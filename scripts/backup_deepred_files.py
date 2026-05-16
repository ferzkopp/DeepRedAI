#!/usr/bin/env python3
"""
Backup selected DeepRedAI corpus files to a remote server over SSH/SFTP.

Current scope:
  - $CHESS_DATA/corpus/chess_games.jsonl.gz
  - $CHESS_DATA/corpus/augmented_chess_games.jsonl.gz

Behavior:
  - Prompts interactively for server, username, password, and target folder.
  - Uploads files to the remote folder, overwriting existing files.
  - Shows per-file upload progress.
  - Saves non-secret config locally for subsequent runs.
  - Attempts secure password storage via keyring when available.

Examples:
  source /mnt/data/DeepRedAI/deepred-env.sh
  python3 scripts/backup_deepred_files.py
  python3 scripts/backup_deepred_files.py --host backup.example.com --username alice
  python3 scripts/backup_deepred_files.py --dry-run
  python3 scripts/backup_deepred_files.py --save-password insecure
  python3 scripts/backup_deepred_files.py --reset-config
"""

import argparse
import getpass
import json
import logging
import os
import posixpath
import stat
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import paramiko
except ImportError:
    paramiko = None

try:
    import keyring
except ImportError:
    keyring = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

# Keep Paramiko's verbose transport/auth chatter out of normal INFO output.
logging.getLogger('paramiko').setLevel(logging.WARNING)
logging.getLogger('paramiko.transport').setLevel(logging.WARNING)


CHESS_DATA = Path(
    os.environ.get(
        'CHESS_DATA',
        os.path.join(os.environ.get('DEEPRED_ROOT', '/mnt/data'), 'chess'),
    )
)
CORPUS_DIR = CHESS_DATA / 'corpus'
FILES_TO_BACKUP = [
    CORPUS_DIR / 'chess_games.jsonl.gz',
    CORPUS_DIR / 'augmented_chess_games.jsonl.gz',
]

DEFAULT_TARGET_FOLDER = '/Data'
DEFAULT_PORT = 22
KEYRING_SERVICE_NAME = 'deepredai-backup-upload'

CONFIG_DIR = Path.home() / '.config' / 'deepredai'
CONFIG_FILE = CONFIG_DIR / 'backup_upload.json'


def _prompt_with_default(prompt: str, default: Optional[str]) -> str:
    if default:
        value = input(f"{prompt} [{default}]: ").strip()
        return value or default
    return input(f"{prompt}: ").strip()


def _prompt_port(default_port: int) -> int:
    while True:
        value = _prompt_with_default('SSH port', str(default_port))
        try:
            port = int(value)
            if port <= 0:
                raise ValueError()
            return port
        except ValueError:
            log.warning('Invalid SSH port: %s (must be a positive integer)', value)


def _normalize_remote_folder(value: str) -> str:
    value = value.strip()
    if not value:
        return DEFAULT_TARGET_FOLDER
    if not value.startswith('/'):
        return '/' + value
    return value


def _load_config() -> Dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not read config file %s: %s", CONFIG_FILE, e)
        return {}


def _save_config(config: Dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CONFIG_FILE.with_suffix('.json.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
        f.write('\n')
    os.chmod(tmp_path, 0o600)
    tmp_path.replace(CONFIG_FILE)


def _reset_config():
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
        log.info("Deleted config file: %s", CONFIG_FILE)


def _build_keyring_account(host: str, username: str, port: int) -> str:
    return f"{username}@{host}:{port}"


def _load_password_from_keyring(host: str, username: str, port: int) -> Optional[str]:
    if keyring is None:
        return None
    account = _build_keyring_account(host, username, port)
    try:
        return keyring.get_password(KEYRING_SERVICE_NAME, account)
    except Exception as e:
        log.debug("Keyring read failed for %s: %s", account, e)
        return None


def _save_password_to_keyring(host: str, username: str, port: int, password: str) -> bool:
    if keyring is None:
        return False
    account = _build_keyring_account(host, username, port)
    try:
        keyring.set_password(KEYRING_SERVICE_NAME, account, password)
        return True
    except Exception as e:
        log.warning("Could not store password in keyring (%s): %s", account, e)
        return False


def _delete_password_from_keyring(host: str, username: str, port: int):
    if keyring is None:
        return
    account = _build_keyring_account(host, username, port)
    try:
        keyring.delete_password(KEYRING_SERVICE_NAME, account)
    except Exception:
        pass


def _ensure_remote_dir(sftp, remote_dir: str):
    parts = [p for p in remote_dir.split('/') if p]
    current = '/'
    for part in parts:
        current = posixpath.join(current, part)
        try:
            st = sftp.stat(current)
            if not stat.S_ISDIR(st.st_mode):
                raise RuntimeError(f"Remote path exists but is not a directory: {current}")
        except FileNotFoundError:
            sftp.mkdir(current)
        except IOError:
            try:
                st = sftp.stat(current)
                if not stat.S_ISDIR(st.st_mode):
                    raise RuntimeError(f"Remote path exists but is not a directory: {current}")
            except Exception as e:
                raise RuntimeError(f"Could not create remote directory {current}: {e}") from e


class _UploadProgress:
    def __init__(self, filename: str, total: int):
        self.filename = filename
        self.total = total
        self._last_pct = -1
        self._bar = None
        self._closed = False
        if tqdm is not None:
            self._bar = tqdm(
                total=total,
                unit='B',
                unit_scale=True,
                desc=filename,
                leave=True,
            )

    def callback(self, transferred: int, total: int):
        if self._bar is not None:
            delta = transferred - self._bar.n
            if delta > 0:
                self._bar.update(delta)
            return

        if total <= 0:
            return
        pct = int((transferred * 100) / total)
        if pct != self._last_pct and pct % 5 == 0:
            self._last_pct = pct
            print(f"{self.filename}: {pct}%", flush=True)

    def close(self):
        if self._closed:
            return
        if self._bar is not None:
            self._bar.close()
        self._closed = True


def _gather_credentials(args, config: Dict) -> Tuple[str, str, int, str, str]:
    has_saved_settings = any(
        key in config for key in ('host', 'username', 'port', 'target_folder')
    )
    if has_saved_settings:
        log.info('Loaded saved backup settings from %s', CONFIG_FILE)
        log.info('Press Enter to keep current values, or type new values to edit.')

    host_default = config.get('host', '')
    username_default = config.get('username', getpass.getuser())
    port_default = config.get('port', DEFAULT_PORT)
    target_default = config.get('target_folder', DEFAULT_TARGET_FOLDER)

    if args.host:
        host = args.host
    else:
        host = _prompt_with_default(
            'Backup server hostname or IP',
            host_default or None,
        )

    if args.username:
        username = args.username
    else:
        username = _prompt_with_default('SSH username', username_default)

    if args.port is not None:
        port = args.port
    else:
        port = _prompt_port(int(port_default))

    if args.target_folder:
        target_folder = _normalize_remote_folder(args.target_folder)
    else:
        target_folder = _normalize_remote_folder(
            _prompt_with_default('Remote target folder', target_default)
        )

    password = args.password
    if not password:
        password = _load_password_from_keyring(host, username, port)
        if password:
            log.info('Using saved password from secure keyring.')

    if not password:
        password = getpass.getpass('SSH password: ')

    if not host or not username or not password:
        raise ValueError('Host, username, and password are required.')

    return host, username, port, target_folder, password


def _validate_local_files() -> bool:
    missing = [p for p in FILES_TO_BACKUP if not p.exists()]
    if missing:
        for path in missing:
            log.error('Missing required backup file: %s', path)
        log.error('Create compressed files first: python3 scripts/augment_chess_games.py --compress')
        return False
    return True


def _save_runtime_config(args, host: str, username: str, port: int,
                         target_folder: str, password: str, config: Dict):
    if args.no_save_config:
        return

    config['host'] = host
    config['username'] = username
    config['port'] = port
    config['target_folder'] = target_folder

    save_mode = args.save_password

    if save_mode == 'never':
        config.pop('insecure_password', None)
        _delete_password_from_keyring(host, username, port)
    elif save_mode == 'insecure':
        config['insecure_password'] = password
        _delete_password_from_keyring(host, username, port)
        log.warning('Password stored insecurely in local config file: %s', CONFIG_FILE)
    else:  # auto
        config.pop('insecure_password', None)
        saved = _save_password_to_keyring(host, username, port, password)
        if not saved:
            log.info('Secure keyring storage unavailable. Password will be requested on next run.')

    _save_config(config)
    log.info('Saved backup configuration to %s', CONFIG_FILE)


def _connect_sftp(host: str, username: str, password: str, port: int):
    if paramiko is None:
        log.error('Missing dependency: paramiko')
        log.error('Install with: pip install paramiko')
        sys.exit(1)

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=username,
        password=password,
        timeout=20,
        allow_agent=False,
        look_for_keys=False,
    )
    return client, client.open_sftp()


def _upload_files(sftp, target_folder: str) -> Tuple[int, int]:
    uploaded = 0
    failed = 0

    _ensure_remote_dir(sftp, target_folder)

    for local_file in FILES_TO_BACKUP:
        local_size = local_file.stat().st_size
        remote_final = posixpath.join(target_folder, local_file.name)
        remote_tmp = posixpath.join(target_folder, f".{local_file.name}.uploading")

        log.info('Uploading %s -> %s', local_file, remote_final)
        progress = _UploadProgress(local_file.name, local_size)

        try:
            sftp.put(str(local_file), remote_tmp, callback=progress.callback)
            try:
                sftp.remove(remote_final)
            except IOError:
                pass
            sftp.rename(remote_tmp, remote_final)
            remote_size = sftp.stat(remote_final).st_size
            if remote_size != local_size:
                raise RuntimeError(
                    f"Size mismatch after upload ({local_size} != {remote_size})"
                )
            uploaded += 1
            progress.close()
            log.info('Uploaded %s (%d bytes)', local_file.name, local_size)
        except Exception as e:
            progress.close()
            failed += 1
            log.error('Failed to upload %s: %s', local_file.name, e)
            try:
                sftp.remove(remote_tmp)
            except Exception:
                pass
        finally:
            progress.close()

    return uploaded, failed


def main():
    parser = argparse.ArgumentParser(
        description='Upload DeepRedAI backup files to a remote SSH/SFTP server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--host', default=None,
                        help='SSH server hostname or IP')
    parser.add_argument('--username', default=None,
                        help='SSH username')
    parser.add_argument('--port', type=int, default=None,
                        help=f'SSH port (default: {DEFAULT_PORT})')
    parser.add_argument('--password', default=None,
                        help='SSH password (discouraged on CLI; prompts if omitted)')
    parser.add_argument('--target-folder', default=None,
                        help=f'Remote folder (default: {DEFAULT_TARGET_FOLDER})')
    parser.add_argument('--save-password', choices=['auto', 'never', 'insecure'],
                        default='auto',
                        help='Password storage mode: auto=keyring if available, '
                             'never=do not store, insecure=store in local config file')
    parser.add_argument('--no-save-config', action='store_true',
                        help='Do not persist host/user/target config')
    parser.add_argument('--reset-config', action='store_true',
                        help='Delete saved local config and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be uploaded without connecting')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)
        # Re-enable Paramiko diagnostics only in verbose mode.
        logging.getLogger('paramiko').setLevel(logging.INFO)
        logging.getLogger('paramiko.transport').setLevel(logging.INFO)

    if args.reset_config:
        _reset_config()
        log.info('Config reset complete')
        return

    if not _validate_local_files():
        sys.exit(1)

    config = _load_config()

    if 'insecure_password' in config and not args.password:
        log.warning('Using insecure password stored in local config file.')
        args.password = config.get('insecure_password')

    try:
        host, username, port, target_folder, password = _gather_credentials(args, config)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    log.info('Backup source files:')
    for f in FILES_TO_BACKUP:
        log.info('  - %s', f)
    log.info('Remote destination: %s@%s:%s%s', username, host, port, target_folder)

    if args.dry_run:
        log.info('Dry-run mode: no upload performed')
        _save_runtime_config(args, host, username, port, target_folder, password, config)
        return

    client = None
    sftp = None
    
    # Attempt connection, with retry on auth failure
    try:
        client, sftp = _connect_sftp(host, username, password, port)
    except Exception as e:
        # Check if this is an authentication error we can retry
        is_auth_error = False
        if paramiko is not None:
            try:
                is_auth_error = isinstance(e, paramiko.ssh_exception.AuthenticationException)
            except AttributeError:
                is_auth_error = False
        
        if is_auth_error:
            log.warning('Authentication failed.')
            log.info('Deleting failed password from secure storage.')
            _delete_password_from_keyring(host, username, port)
            
            password = getpass.getpass('SSH password (retry): ')
            if not password:
                log.error('Password required to retry.')
                sys.exit(1)
            
            log.info('Retrying connection with new password...')
            try:
                client, sftp = _connect_sftp(host, username, password, port)
            except KeyboardInterrupt:
                log.error('Interrupted by user')
                sys.exit(130)
            except Exception as retry_err:
                log.error('Connection failed on retry: %s', retry_err)
                sys.exit(1)
        elif isinstance(e, KeyboardInterrupt):
            log.error('Interrupted by user')
            sys.exit(130)
        else:
            log.error('Connection failed: %s', e)
            sys.exit(1)
    except KeyboardInterrupt:
        log.error('Interrupted by user')
        sys.exit(130)
    
    # Upload files
    try:
        uploaded, failed = _upload_files(sftp, target_folder)
    except KeyboardInterrupt:
        log.error('Interrupted by user')
        sys.exit(130)
    except Exception as e:
        log.error('Backup failed: %s', e)
        sys.exit(1)
    finally:
        try:
            sftp.close()
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass

    _save_runtime_config(args, host, username, port, target_folder, password, config)

    log.info('')
    log.info('Backup completed')
    log.info('  Uploaded: %d', uploaded)
    log.info('  Failed: %d', failed)

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
