import logging
import logging.config
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Đảm bảo thư mục logs tồn tại
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Cấu hình logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "verbose": {
            "format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_dir / "app.log",  # Đặt log file ở thư mục logs
            "formatter": "verbose",
            "level": "DEBUG",
            "maxBytes": 5 * 1024 * 1024,  # 5 MB mỗi file log
            "backupCount": 5,  # Lưu trữ tối đa 5 file log cũ
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "app_logger": {  # specific logger for the app
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app_logger")
