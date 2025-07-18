import asyncio

async def fetch_data(delay : int, id : int) -> dict:
    print("Fetching data...id:", id)
    await asyncio.sleep(delay)
    print("Data fetched, id:", id)
    return {"data":"Some data", "id": id}

#Simple example of using asyncio to run multiple coroutines concurrently
async def main():
    print("Starting main coroutine...")
    task1 = asyncio.create_task(fetch_data(2, 1))
    task2 = asyncio.create_task(fetch_data(3, 2))
    task3 = asyncio.create_task(fetch_data(1, 3))
    print("End of main coroutine, waiting for task to complete...")

    result1 = await task1
    print(f"Received result: {result1}")

    result2 = await task2
    print(f"Received result: {result2}")

    result3 = await task3
    print(f"Received result: {result3}")    

asyncio.run(main())


#Simple example of using gather to run multiple coroutines concurrently
async def main():
    print("Starting main coroutine...")
    results = await asyncio.gather(fetch_data(2, 1), fetch_data(3, 2), fetch_data(1, 3))
    print("End of main coroutine, waiting for task to complete...")

    for result in results:
        print(f"Received result: {result}")

asyncio.run(main())


#Simple example of using taskgroup (has built in error handling as it will break all tasks if one fails)
async def main():
    tasks = []
    async with asyncio.TaskGroup() as tg:
        for i, sleep_time in enumerate([2, 1, 3], start = 1):
            task = tg.create_task(fetch_data(sleep_time, i))
            tasks.append(task)
    
    results = [task.result() for task in tasks]
    for result in results:
        print(f"Received result: {result}")

asyncio.run(main())