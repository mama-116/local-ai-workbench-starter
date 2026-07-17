from __future__ import annotations

import flet as ft

from local_llm_chat.bootstrap import bootstrap
from local_llm_chat.presentation.flet_app import LocalChatApp
from local_llm_chat.infrastructure.process_restart import signal_restart_ready


async def main(page: ft.Page) -> None:
    container = await bootstrap()

    async def close_container() -> None:
        await container.close()

    page.on_disconnect = close_container
    app = LocalChatApp(page, container)
    await app.initialize()
    signal_restart_ready(container.paths.data_dir)


if __name__ == "__main__":
    ft.run(main)
