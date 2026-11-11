import asyncio

from bleak import BleakClient, BleakScanner

UUID_COMANDOS = "b5f90072-aa8d-11e3-9046-0002a5d5c51b"
UUID_RESPUESTA = "b5f90073-aa8d-11e3-9046-0002a5d5c51b"
UUID_RED = "b5f90002-aa8d-11e3-9046-0002a5d5c51b"
UUID_CLAVE = "b5f90003-aa8d-11e3-9046-0002a5d5c51b"


async def principal():
    print("Buscando GoPro por Bluetooth...")
    dispositivo = await BleakScanner.find_device_by_filter(
        lambda d, ad: d.name is not None and d.name.startswith("GoPro")
    )
    if dispositivo is None:
        print("No se encontró la cámara. Ponla en modo emparejamiento.")
        return

    async with BleakClient(dispositivo, timeout=30) as cliente:
        try:
            await cliente.pair()
        except Exception:
            pass

        red = (await cliente.read_gatt_char(UUID_RED)).decode()
        clave = (await cliente.read_gatt_char(UUID_CLAVE)).decode()

        await cliente.start_notify(UUID_RESPUESTA, lambda _, datos: None)
        await cliente.write_gatt_char(
            UUID_COMANDOS, bytearray([0x03, 0x17, 0x01, 0x01]), response=True
        )
        await asyncio.sleep(2)

        print("Red:", red)
        print("Clave:", clave)
        input("Conecta el PC a esa red WiFi y presiona Enter...")


asyncio.run(principal())