import json
from pathlib import Path
from datetime import datetime

class SettingsManager:
    """アプリケーション設定の管理クラス"""
    
    def __init__(self):
        self.settings_dir = Path.home() / ".yolo_mosaic_app"
        self.settings_file = self.settings_dir / "settings.json"
        self.recent_datasets_file = self.settings_dir / "recent_datasets.json"
        
        # 設定ディレクトリを作成
        self.settings_dir.mkdir(exist_ok=True)
        
        # デフォルト設定
        self.default_settings = {
            "last_dataset": None,
            "window_geometry": None,
            "last_tab_index": 0,
            "auto_save": True,
            "max_recent_datasets": 10
        }
        
        self.settings = self.load_settings()
        self.recent_datasets = self.load_recent_datasets()
    
    def load_settings(self):
        """設定ファイルを読み込む"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                # デフォルト設定とマージ
                return {**self.default_settings, **settings}
            except Exception as e:
                print(f"設定ファイル読み込みエラー: {e}")
        
        return self.default_settings.copy()
    
    def save_settings(self):
        """設定ファイルを保存"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"設定ファイル保存エラー: {e}")
    
    def load_recent_datasets(self):
        """最近使用したデータセットのリストを読み込む"""
        if self.recent_datasets_file.exists():
            try:
                with open(self.recent_datasets_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"最近のデータセット読み込みエラー: {e}")
        
        return []
    
    def save_recent_datasets(self):
        """最近使用したデータセットのリストを保存"""
        try:
            with open(self.recent_datasets_file, 'w', encoding='utf-8') as f:
                json.dump(self.recent_datasets, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"最近のデータセット保存エラー: {e}")
    
    def add_recent_dataset(self, dataset_path):
        """最近使用したデータセットを追加"""
        dataset_info = {
            "path": str(dataset_path),
            "name": Path(dataset_path).name,
            "last_used": datetime.now().isoformat()
        }
        
        # 既存のエントリを削除
        self.recent_datasets = [
            d for d in self.recent_datasets 
            if d["path"] != str(dataset_path)
        ]
        
        # 先頭に追加
        self.recent_datasets.insert(0, dataset_info)
        
        # 最大数を超えたら削除
        max_recent = self.settings.get("max_recent_datasets", 10)
        self.recent_datasets = self.recent_datasets[:max_recent]
        
        self.save_recent_datasets()
    
    def get_recent_datasets(self):
        """最近使用したデータセットのリストを取得"""
        # 存在しないパスを除外
        valid_datasets = []
        for dataset in self.recent_datasets:
            if Path(dataset["path"]).exists():
                valid_datasets.append(dataset)
        
        # 更新されたリストを保存
        if len(valid_datasets) != len(self.recent_datasets):
            self.recent_datasets = valid_datasets
            self.save_recent_datasets()
        
        return valid_datasets
    
    def get_last_dataset(self):
        """最後に使用したデータセットのパスを取得"""
        return self.settings.get("last_dataset")
    
    def save_last_dataset(self, dataset_path):
        """最後に使用したデータセットのパスを保存"""
        self.settings["last_dataset"] = str(dataset_path)
        self.save_settings()
        
        # 最近使用したデータセットにも追加
        self.add_recent_dataset(dataset_path)
    
    def get_window_geometry(self):
        """ウィンドウのジオメトリを取得"""
        return self.settings.get("window_geometry")
    
    def save_window_geometry(self, geometry):
        """ウィンドウのジオメトリを保存"""
        self.settings["window_geometry"] = geometry
        self.save_settings()
    
    def get_last_tab_index(self):
        """最後に開いていたタブのインデックスを取得"""
        return self.settings.get("last_tab_index", 0)
    
    def save_last_tab_index(self, index):
        """最後に開いていたタブのインデックスを保存"""
        self.settings["last_tab_index"] = index
        self.save_settings()
    
    def is_auto_save_enabled(self):
        """自動保存が有効かどうか"""
        return self.settings.get("auto_save", True)
    
    def set_auto_save(self, enabled):
        """自動保存の有効/無効を設定"""
        self.settings["auto_save"] = enabled
        self.save_settings()