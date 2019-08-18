import config
import serial


class ArduinoController(object):

    def __init__(self, SERIAL_PORT_PATH, BAUDRATE):

        self.SERIAL_PORT_PATH = SERIAL_PORT_PATH
        self.BAUDRATE = BAUDRATE
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

    # arduino 1 (uno for camera)
    def start_pulses(self, numFramesToAcquire):
        command = "PULSE " + str(numFramesToAcquire) + " " + str(config.TRIGGER_FREQUENCY_US) + "\n"
        self.serialInterface.write(command.encode('UTF-8'))
        # self.serialInterface.readline().decode()
        # return True

        # arduino 2 (mega for motors)
    def move_motor_steps(self, motorIndex, steps, speed):
        steps = int(steps)
        command = "MOVE S" + str(motorIndex) + " " + str(steps) + " " + str(speed) + "\n"
        self.serialInterface.write(command.encode('UTF-8'))
        response = self.serialInterface.readline().decode()
        print(response)

