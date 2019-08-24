from multiprocessing import Process, Queue
from FlirController import *
from ArduinoController import *
from PS4Controller import *


def launch_system_processes():
    queues = [Queue(0), Queue(0), Queue(0), Queue(0)]
    processes = []

    processes.append(Process(target=launch_arduino_controller, args=(queues[1], queues[0], 'COM6', 115200,)))
    processes.append(Process(target=launch_ps4_controller, args=(queues[2], queues[0],)))
    processes.append(Process(target=launch_camera_controller, args=(queues[3], queues[0],)))

    for process in processes:
        process.start()
    return processes, queues


def main():

    processes, queues = launch_system_processes()

    while True:
        if not queues[0].empty():
            msg = queues[0].get()
            queues[msg[0]].put(msg)


if __name__ == '__main__':
    main()
