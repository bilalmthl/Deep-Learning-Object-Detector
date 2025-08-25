import time, subprocess, requests
from pathlib import Path
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD

# Config
SWITCH_PIN = 17   
SERVER_URL = "http://10.0.0.107:8000/predict"   
IMG_PATH = Path("/home/bilal/capture.jpg")

# LCD setup
lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)

def lcd_message(line1, line2=""):
    lcd.clear()
    lcd.write_string(line1[:16])
    if line2:
        lcd.crlf()
        lcd.write_string(line2[:16])

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def capture_jpeg(out_path):
    """Capture a single JPEG using raspistill."""
    out_path = str(out_path)
    try:
        subprocess.run([
            "raspistill", "-n", "-t", "1000",
            "-o", out_path, "-q", "85", "-w", "640", "-h", "480"
        ], check=True)
        return True
    except Exception as e:
        print("raspistill error:", e)
        return False

def post_for_prediction(path):
    with open(path, "rb") as f:
        r = requests.post(SERVER_URL, files={"image": f}, timeout=20)
    r.raise_for_status()
    return r.json()

DISPLAY_MAP = {"apples": "APPLE", "bananas": "BANANA", "oranges": "ORANGE", "none": "NONE"}

def main():
    last_state = GPIO.input(SWITCH_PIN)
    lcd_message("Ready", "Flip to start")
    try:
        while True:
            state = GPIO.input(SWITCH_PIN)

            # Switch goes HIGH
            if state != last_state and state == GPIO.HIGH:
                lcd_message("DETECTING...")
                ok = capture_jpeg(IMG_PATH)
                if not ok:
                    lcd_message("CAMERA ERROR")
                else:
                    try:
                        data = post_for_prediction(IMG_PATH)
                        label = data.get("label", "none")
                        display = DISPLAY_MAP.get(label, label.upper())
                        lcd_message(display, "TURN SWITCH OFF")
                    except Exception as e:
                        print("Server error:", e)
                        lcd_message("SERVER ERROR")

                # Wait until user turns switch off
                while GPIO.input(SWITCH_PIN) == GPIO.HIGH:
                    time.sleep(0.05)

                lcd_message("Ready", "Flip to start")

            last_state = state
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        lcd.clear()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
