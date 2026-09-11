"""夹取日志开关：默认安静，排障时打开 verbose_logs / pinch_debug。"""
from __future__ import annotations


def eye_verbose(eye) -> bool:
    if not isinstance(eye, dict):
        return False
    return bool(eye.get("verbose_logs", False))


def eye_debug(eye) -> bool:
    """几何/TCP/MJ→robot 细节；verbose 也打开 debug。"""
    if not isinstance(eye, dict):
        return False
    return bool(eye.get("pinch_debug", False)) or eye_verbose(eye)


def dbg(log, eye, msg):
    if log and eye_debug(eye):
        log(msg)


def vrb(log, eye, msg):
    if log and eye_verbose(eye):
        log(msg)


def move_print(eye, msg, *, important=False):
    """[move] 通道：默认只打失败/中止；成功进度走 verbose。"""
    if important or eye_verbose(eye):
        print(msg, flush=True)
