import time,asyncio

async def count():
    print("count one")
    await asyncio.sleep(1)
    print("count four")

async def count_further():
    print("count two")
    await asyncio.sleep(1)
    print("count five")

async def count_even_further():
    print("count three")
    await asyncio.sleep(1)
    print("count six")

async def main():
    await asyncio.gather(count(), count_further(), count_even_further())

if __name__ == '__main__':
    s = time.perf_counter()
    await main()
    elapsed = time.perf_counter() - s
    print(f"Script executed in {elapsed:0.2f} seconds.")
