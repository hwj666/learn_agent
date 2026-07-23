import asyncio
from tracing.infra.exporter import BatchExporter
from tracing.transports.noop import NoopTransport


async def main():
    exporter = BatchExporter(
        transport=NoopTransport(),
        serializer=lambda event: event,
        batch_size=10,  # 立即刷新
        schedule_delay=0.1,  # 更快响应
    )

    await exporter.start()
    await exporter.export({"he": "wen1"})

    await exporter.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
