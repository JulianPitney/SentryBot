import config
import serial


class ArduinoController(object):

    def __init__(self, queue, mainQueue, SERIAL_PORT_PATH, BAUDRATE):

        self.SERIAL_PORT_PATH = SERIAL_PORT_PATH
        self.BAUDRATE = BAUDRATE
        self.queue = queue
        self.mainQueue = mainQueue
        # Motor configuration
        self.SEEK_SPEED = 8000
        self.JOG_MIN_SPEED = 800
        self.JOG_MAX_SPEED = 2000
        self.MICROMETERS_PER_STEP = 0.15625
        self.serialInterface = self.open_serial_interface()
        self.wait_for_arduino_confirmation()

    def __del__(self):
        if self.serialInterface != None:
            self.serialInterface.close()

    def open_serial_interface(self):

        # Try to open serial connection until it works
        while (1):
            serialInterface = serial.Serial(self.SERIAL_PORT_PATH, self.BAUDRATE, timeout=3)
            if serialInterface.is_open:
                return serialInterface
            else:
                print("Unable to open serial interface!")
                return None

    def wait_for_arduino_confirmation(self):

        # Wait for arduino to say it's ready
        while (1):
            confirmation = self.serialInterface.readline().decode()
            if confirmation == "ARDUINO READY\n":
                print(confirmation)
                break
            else:
                print("No response from arduino!")
                break


    def map_analog_to_discrete_range(self, value, leftMin, leftMax, rightMin, rightMax):

        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        valueScaled = float(value - leftMin) / float(leftSpan)
        return int(rightMin + (valueScaled * rightSpan))

    def toggle_coarse_jog(self):

        command = "TOGGLE_COARSE_JOG\n"
        self.serialInterface.write(command.encode('UTF-8'))
        response = self.serialInterface.readline().decode()
        print(response)


    # arduino 1 (uno for camera)
    def start_pulses(self, numFramesToAcquire):
        command = "PULSE " + str(numFramesToAcquire) + " " + str(config.TRIGGER_FREQUENCY_US) + "\n"
        self.serialInterface.write(command.encode('UTF-8'))
        #self.serialInterface.readline().decode()



    def toggle_led(self):

        command = "TOGGLE_LED\n"
        self.serialInterface.write(command.encode('UTF-8'))
        response = self.serialInterface.readline().decode()
        print(response)



    def jog_motor(self, motorInputs):
        speeds = []

        for motorInput in motorInputs:

            if motorInput > 0.3:
                speed = self.map_analog_to_discrete_range(motorInput, 0.3, 1, self.JOG_MIN_SPEED, self.JOG_MAX_SPEED)
            elif motorInput < -0.3:
                speed = self.map_analog_to_discrete_range(motorInput, -0.3, -1, -self.JOG_MIN_SPEED, -self.JOG_MAX_SPEED)
            else:
                speed = 0

            speeds.append(speed)

        command = "JOG " + str(speeds[0]) + " " + str(speeds[1]) + " " + str(speeds[2]) + "\n"
        self.serialInterface.write(command.encode('UTF-8'))
        response = self.serialInterface.readline().decode()
        print(response)

    def process_msg(self, msg):
        funcIndex = msg[1]

        if funcIndex == 4:
            self.jog_motor(msg[2])
        elif funcIndex == 5:
            self.toggle_coarse_jog()
        elif funcIndex == 8:
            self.toggle_led()


    def mainloop(self):
        while True:
            if not self.queue.empty():
                self.process_msg(self.queue.get())


def launch_arduino_controller(queue, mainQueue, SERIAL_PORT_PATH, BAUDRATE):

    ac = ArduinoController(queue, mainQueue, SERIAL_PORT_PATH, BAUDRATE)
    ac.mainloop()