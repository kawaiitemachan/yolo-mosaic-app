import os
import sys
from pathlib import Path

def get_application_path():
    """
    実行ファイルまたは開発環境でのベースパスを返す
    PyInstallerでパッケージ化された場合とそうでない場合の両方に対応
    """
    if getattr(sys, 'frozen', False):
        # PyInstallerでパッケージ化された場合
        # --onefileの場合、sys.executable は実行ファイルのパス
        return Path(sys.executable).parent
    else:
        # 開発環境での実行
        # main.pyのあるディレクトリを返す
        return Path(__file__).parent.parent.parent

def get_resource_path(relative_path):
    """
    リソースファイルの絶対パスを返す
    PyInstallerの--onefileオプションで埋め込まれたファイルに対応
    """
    if getattr(sys, 'frozen', False):
        # PyInstallerでパッケージ化された場合
        # sys._MEIPASSは一時的に展開されたファイルのディレクトリ
        base_path = Path(sys._MEIPASS)
    else:
        # 開発環境での実行
        base_path = Path(__file__).parent.parent.parent
    
    return base_path / relative_path

def ensure_directories():
    """
    アプリケーションに必要なディレクトリを作成する
    実行ファイルと同じ階層に作成される
    """
    app_path = get_application_path()
    
    # 必要なディレクトリのリスト
    directories = [
        'data',
        'data/images',
        'data/annotations',
        'data/models',
        'datasets',
    ]
    
    # ディレクトリの作成
    for directory in directories:
        dir_path = app_path / directory
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {dir_path}")
            except Exception as e:
                print(f"Failed to create directory {dir_path}: {e}")
    
    return app_path

def get_data_path(subpath=""):
    """
    データディレクトリのパスを返す
    """
    app_path = get_application_path()
    data_path = app_path / "data"
    
    if subpath:
        return data_path / subpath
    return data_path

def get_datasets_path(subpath=""):
    """
    データセットディレクトリのパスを返す
    """
    app_path = get_application_path()
    datasets_path = app_path / "datasets"
    
    if subpath:
        return datasets_path / subpath
    return datasets_path

def get_models_path(subpath=""):
    """
    モデルディレクトリのパスを返す
    """
    return get_data_path(f"models/{subpath}" if subpath else "models")