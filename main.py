import config
import FlirController as fc
import ArduinoController as ac

def main():

    flirController = fc.CameraController()
    arduino_motors = ac.ArduinoController('COM6', 115200)

    while True:

        menuSelection = input("[1] ACTIVATE SENTRY\nInput: ")
        try:
           menuSelection = int(menuSelection)
        except ValueError:
            print("Invalid selection, try again.")
            continue

        if menuSelection == 1:
            flirController.synchronous_record()

        elif menuSelection == 2:
            steps = int(input("Steps: "))

            arduino_motors.move_motor_steps(1,steps, 4000)
            arduino_motors.move_motor_steps(2, steps, 4000)
            arduino_motors.move_motor_steps(1, -steps, 4000)
            arduino_motors.move_motor_steps(2, -steps, 4000)



if __name__ == '__main__':
	main()


