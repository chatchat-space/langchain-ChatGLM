import asyncio
import logging
import logging.config
import multiprocessing as mp
import os
import sys
from contextlib import asynccontextmanager
from multiprocessing import Process

# 设置numexpr最大线程数，默认为CPU核心数
try:
    import numexpr

    n_cores = numexpr.utils.detect_number_of_cores()
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass

import click
from typing import Dict, List

from fastapi import FastAPI

from chatchat.utils import build_logger


logger = build_logger()


def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if started_event is not None:
            started_event.set()
        yield

    app.router.lifespan_context = lifespan


def run_api_server(
    started_event: mp.Event = None, run_mode: str = None
):
    import uvicorn
    from chatchat.utils import (
        get_config_dict,
        get_log_file,
        get_timestamp_ms,
    )

    from chatchat.settings import Settings
    from chatchat.server.api_server.server_app import create_app
    from chatchat.server.utils import set_httpx_config, set_graph_memory

    logger.info(f"Api MODEL_PLATFORMS: {Settings.model_settings.MODEL_PLATFORMS}")
    set_httpx_config()
    app = create_app(run_mode=run_mode)
    _set_app_event(app, started_event)

    host = Settings.basic_settings.API_SERVER["host"]
    port = Settings.basic_settings.API_SERVER["port"]

    logging_conf = get_config_dict(
        "INFO",
        get_log_file(log_path=Settings.basic_settings.LOG_PATH, sub_dir=f"run_api_server_{get_timestamp_ms()}"),
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    logging.config.dictConfig(logging_conf)  # type: ignore

    set_graph_memory()

    if Settings.basic_settings.LANG_SMITH["OPEN_LANGSMITH"] is True:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = Settings.basic_settings.LANG_SMITH["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_PROJECT"] = Settings.basic_settings.LANG_SMITH["LANGCHAIN_PROJECT"]
    logger.info(f"LANG_SMITH CONFIG: {Settings.basic_settings.LANG_SMITH}")

    uvicorn.run(app, host=host, port=port)


def run_webui(
    started_event: mp.Event = None, run_mode: str = None
):
    from chatchat.settings import Settings
    from chatchat.server.utils import set_httpx_config
    from chatchat.utils import get_config_dict, get_log_file, get_timestamp_ms

    logger.info(f"Webui MODEL_PLATFORMS: {Settings.model_settings.MODEL_PLATFORMS}")
    set_httpx_config()

    host = Settings.basic_settings.WEBUI_SERVER["host"]
    port = Settings.basic_settings.WEBUI_SERVER["port"]

    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui.py")

    flag_options = {
        "server_address": host,
        "server_port": port,
        "theme_base": "light",
        "theme_primaryColor": "#165dff",
        "theme_secondaryBackgroundColor": "#f5f5f5",
        "theme_textColor": "#000000",
        "global_disableWatchdogWarning": None,
        "global_disableWidgetStateDuplicationWarning": None,
        "global_showWarningOnDirectExecution": None,
        "global_developmentMode": None,
        "global_logLevel": None,
        "global_unitTest": None,
        "global_suppressDeprecationWarnings": None,
        "global_minCachedMessageSize": None,
        "global_maxCachedMessageAge": None,
        "global_storeCachedForwardMessagesInMemory": None,
        "global_dataFrameSerialization": None,
        "logger_level": None,
        "logger_messageFormat": None,
        "logger_enableRich": None,
        "client_caching": None,
        "client_displayEnabled": None,
        "client_showErrorDetails": None,
        "client_toolbarMode": None,
        "client_showSidebarNavigation": None,
        "runner_magicEnabled": None,
        "runner_installTracer": None,
        "runner_fixMatplotlib": None,
        "runner_postScriptGC": None,
        "runner_fastReruns": None,
        "runner_enforceSerializableSessionState": None,
        "runner_enumCoercion": None,
        "server_folderWatchBlacklist": None,
        "server_fileWatcherType": "none",
        "server_headless": None,
        "server_runOnSave": None,
        "server_allowRunOnSave": None,
        "server_scriptHealthCheckEnabled": None,
        "server_baseUrlPath": None,
        "server_enableCORS": None,
        "server_enableXsrfProtection": None,
        "server_maxUploadSize": None,
        "server_maxMessageSize": None,
        "server_enableArrowTruncation": None,
        "server_enableWebsocketCompression": None,
        "server_enableStaticServing": None,
        "browser_serverAddress": None,
        "browser_gatherUsageStats": None,
        "browser_serverPort": None,
        "server_sslCertFile": None,
        "server_sslKeyFile": None,
        "ui_hideTopBar": None,
        "ui_hideSidebarNav": None,
        "magic_displayRootDocString": None,
        "magic_displayLastExprIfNoSemicolon": None,
        "deprecation_showfileUploaderEncoding": None,
        "deprecation_showImageFormat": None,
        "deprecation_showPyplotGlobalUse": None,
        "theme_backgroundColor": None,
        "theme_font": None,
    }

    args = []
    if run_mode == "lite":
        args += [
            "--",
            "lite",
        ]

    try:
        # for streamlit >= 1.12.1
        from streamlit.web import bootstrap
    except ImportError:
        from streamlit import bootstrap

    logging_conf = get_config_dict(
        "INFO",
        get_log_file(log_path=Settings.basic_settings.LOG_PATH, sub_dir=f"run_webui_{get_timestamp_ms()}"),
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    logging.config.dictConfig(logging_conf)  # type: ignore
    bootstrap.load_config_options(flag_options=flag_options)
    bootstrap.run(script_dir, False, args, flag_options)
    started_event.set()


def dump_server_info(after_start=False, args=None):
    import platform

    import langchain

    from chatchat import __version__
    from chatchat.settings import Settings
    from chatchat.server.utils import api_address, webui_address

    print("\n")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print(f"操作系统：{platform.platform()}.")
    print(f"python版本：{sys.version}")
    print(f"项目版本：{__version__}")
    print(f"langchain版本：{langchain.__version__}")
    print(f"数据目录：{Settings.CHATCHAT_ROOT}")
    print("\n")

    print(f"当前使用的分词器：{Settings.kb_settings.TEXT_SPLITTER_NAME}")

    print(f"默认选用的 Embedding 名称： {Settings.model_settings.DEFAULT_EMBEDDING_MODEL}")

    if after_start:
        print("\n")
        print(f"服务端运行信息：")
        if args.api:
            print(f"    Chatchat Api Server: {api_address()}")
        if args.webui:
            print(f"    Chatchat WEBUI Server: {webui_address()}")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print("\n")


async def start_main_server(args):
    import signal
    import time

    from chatchat.utils import (
        get_config_dict,
        get_log_file,
        get_timestamp_ms,
    )

    from chatchat.settings import Settings

    logging_conf = get_config_dict(
        "INFO",
        get_log_file(
            log_path=Settings.basic_settings.LOG_PATH, sub_dir=f"start_main_server_{get_timestamp_ms()}"
        ),
        1024 * 1024 * 1024 * 3,
        1024 * 1024 * 1024 * 3,
    )
    logging.config.dictConfig(logging_conf)  # type: ignore

    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """

        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")

        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn")
    manager = mp.Manager()
    run_mode = None

    if args.all:
        args.api = True
        args.webui = True

    dump_server_info(args=args)

    if len(sys.argv) > 1:
        logger.info(f"正在启动服务：")
        logger.info(f"如需查看 llm_api 日志，请前往 {Settings.basic_settings.LOG_PATH}")

    processes = {}

    def process_count():
        return len(processes)

    api_started = manager.Event()
    if args.api:
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(
                started_event=api_started,
                run_mode=run_mode,
            ),
            daemon=False,
        )
        processes["api"] = process

    webui_started = manager.Event()
    if args.webui:
        process = Process(
            target=run_webui,
            name=f"WEBUI Server",
            kwargs=dict(
                started_event=webui_started,
                run_mode=run_mode,
            ),
            daemon=True,
        )
        processes["webui"] = process

    try:
        if p := processes.get("api"):
            p.start()
            p.name = f"{p.name} ({p.pid})"
            api_started.wait()  # 等待api.py启动完成

        if p := processes.get("webui"):
            p.start()
            p.name = f"{p.name} ({p.pid})"
            webui_started.wait()  # 等待webui.py启动完成

        dump_server_info(after_start=True, args=args)

        # 等待所有进程退出
        while processes:
            for p in processes.values():
                p.join(2)
                if not p.is_alive():
                    processes.pop(p.name)
    except Exception as e:
        logger.error(e)
        logger.warning("Caught KeyboardInterrupt! Setting stop event...")
    finally:
        for p in processes.values():
            logger.warning("Sending SIGKILL to %s", p)
            # Queues and other inter-process communication primitives can break when
            # process is killed, but we don't care here

            if isinstance(p, dict):
                for process in p.values():
                    process.kill()
            else:
                p.kill()

        for p in processes.values():
            logger.info("Process status: %s", p)


@click.command(help="启动服务")
@click.option(
    "-a",
    "--all",
    "all",
    is_flag=True,
    help="run api.py and webui.py",
)
@click.option(
    "--api",
    "api",
    is_flag=True,
    help="run api.py",
)
@click.option(
    "-w",
    "--webui",
    "webui",
    is_flag=True,
    help="run webui.py server",
)
def main(all, api, webui):
    class args:
        ...
    args.all = all
    args.api = api
    args.webui = webui

    # 添加这行代码
    cwd = os.getcwd()
    sys.path.append(cwd)
    mp.freeze_support()
    print("cwd:" + cwd)
    from chatchat.server.knowledge_base.migrate import create_tables

    create_tables()
    if sys.version_info < (3, 10):
        loop = asyncio.get_event_loop()
    else:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
    loop.run_until_complete(start_main_server(args))


if __name__ == "__main__":
    main()
