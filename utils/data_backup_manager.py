import os
import shutil
from datetime import datetime
import zipfile
import sys
import time
import threading
from config import CONFIG


class Spinner:
    def __init__(self, message="Processing"):
        self.spinner = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']
        self.message = message
        self.busy = False
        self.spinner_visible = False
        self.thread = None

    def write_next(self):
        with self._screen_lock:
            if not self.spinner_visible:
                sys.stdout.write(f'\r{self.message} {self.spinner[self.spinner_index]} ')
                self.spinner_index = (self.spinner_index + 1) % len(self.spinner)
                sys.stdout.flush()

    def _spinner_task(self):
        while self.busy:
            self.write_next()
            time.sleep(0.1)
        self.spinner_visible = False

    def __enter__(self):
        if sys.stdout.isatty():
            self._screen_lock = threading.Lock()
            self.busy = True
            self.spinner_index = 0
            self.thread = threading.Thread(target=self._spinner_task)
            self.thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.stdout.isatty():
            self.busy = False
            time.sleep(0.2)
            if self.thread:
                self.thread.join()
            sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
            sys.stdout.flush()


class DataBackupManager:
    def __init__(self):
        self.base_dir = CONFIG.BASE_DIR
        self.rag_data_path = CONFIG.RAG_DATA
        self.backup_dir = os.path.join(self.base_dir, 'backups')

        print("Initializing DataBackupManager...")
        print(f"Base directory: {self.base_dir}")
        print(f"RAG_Data path: {self.rag_data_path}")
        print(f"Backup directory: {self.backup_dir}")

    def create_backup(self):
        """Create a backup of RAG_Data"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f'backup_{timestamp}.zip'
        backup_path = os.path.join(self.backup_dir, backup_filename)

        try:
            with Spinner("Creating backup"):
                with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Backup RAG_Data if it exists
                    if os.path.exists(self.rag_data_path):
                        for root, dirs, files in os.walk(self.rag_data_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, self.base_dir)
                                zipf.write(file_path, arcname)
                    else:
                        print("\nWarning: RAG_Data directory not found")

            backup_size = os.path.getsize(backup_path) / (1024 * 1024)  # Size in MB
            print(f"\nBackup created successfully: {backup_filename} ({backup_size:.2f} MB)")
            return backup_path
        except Exception as e:
            print(f"\nError creating backup: {e}")
            if os.path.exists(backup_path):
                os.remove(backup_path)
            raise

    def restore_backup(self, backup_path):
        """Restore from a backup file"""
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        # Add warning and confirmation
        print("\n⚠️  WARNING: This will overwrite your current files:")
        print("   - All contents in RAG_Data directory will be deleted and replaced")

        confirmation = input("\nAre you sure you want to proceed? (yes/no): ").lower().strip()
        if confirmation != 'yes':
            print("Restore operation cancelled.")
            return

        temp_dir = os.path.join(self.backup_dir, 'temp_restore')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            with Spinner("Restoring backup"):
                # Extract backup
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    zipf.extractall(temp_dir)

                # Restore RAG_Data
                rag_data_backup = os.path.join(temp_dir, 'RAG_Data')
                if os.path.exists(rag_data_backup):
                    if os.path.exists(self.rag_data_path):
                        print("\nRemoving existing RAG_Data directory...")
                        shutil.rmtree(self.rag_data_path)
                    shutil.copytree(rag_data_backup, self.rag_data_path)
                    print("RAG_Data restored successfully")
                else:
                    print("\nWarning: No RAG_Data found in backup")

            print(f"\nBackup restored successfully from: {backup_path}")

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def list_backups(self):
        """List all available backups"""
        if not os.path.exists(self.backup_dir):
            print("No backups found")
            return []

        backups = [f for f in os.listdir(self.backup_dir)
                   if f.startswith('backup_') and f.endswith('.zip')]
        backups.sort(reverse=True)  # Most recent first

        if not backups:
            print("No backups found")
            return []

        print("\nAvailable backups:")
        for backup in backups:
            backup_path = os.path.join(self.backup_dir, backup)
            size = os.path.getsize(backup_path) / (1024 * 1024)  # Convert to MB
            timestamp = backup[7:-4]  # Extract timestamp from filename
            print(f"{backup} ({size:.2f} MB) - {timestamp}")

        return backups


# Example usage
if __name__ == "__main__":
    backup_manager = DataBackupManager()

    while True:
        print("\nRAG Data Backup Manager")
        print("1. Create backup")
        print("2. Restore backup")
        print("3. List backups")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            try:
                backup_path = backup_manager.create_backup()
            except Exception as e:
                print(f"Failed to create backup: {e}")

        elif choice == '2':
            backups = backup_manager.list_backups()
            if backups:
                backup_name = input("Enter the backup filename to restore: ")
                backup_path = os.path.join(backup_manager.backup_dir, backup_name)
                try:
                    backup_manager.restore_backup(backup_path)
                except Exception as e:
                    print(f"Error restoring backup: {e}")

        elif choice == '3':
            backup_manager.list_backups()

        elif choice == '4':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")
