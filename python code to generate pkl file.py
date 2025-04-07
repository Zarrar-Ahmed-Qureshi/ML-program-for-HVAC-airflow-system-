import serial
import time
import matplotlib.pyplot as plt
from collections import deque

PORT = 'COM3'             # Change as needed
BAUD = 9600
MAX_TIME = 10             # seconds of data to show on screen
REFRESH_RATE = 0.1

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
print(f"Connected to {PORT}")

buffer_size = int(MAX_TIME / REFRESH_RATE)

time_buf = deque(maxlen=buffer_size)
x_buf = deque(maxlen=buffer_size)
y_buf = deque(maxlen=buffer_size)
z_buf = deque(maxlen=buffer_size)

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

def update_plot():
    ax1.clear()
    ax2.clear()
    ax3.clear()

    ax1.plot(time_buf, x_buf, 'r')
    ax2.plot(time_buf, y_buf, 'g')
    ax3.plot(time_buf, z_buf, 'b')

    ax1.set_title("X Axis Acceleration")
    ax2.set_title("Y Axis Acceleration")
    ax3.set_title("Z Axis Acceleration")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Accel (m/s²)")
        ax.grid(True)

    plt.tight_layout()
    plt.pause(0.01)

start_time = time.time()
while True:
    try:
        line = ser.readline().decode().strip()
        if not line:
            continue

        parts = line.split(',')
        if len(parts) != 4:
            continue

        t_ms, x, y, z = map(float, parts)
        t = (t_ms / 1000.0) - (start_time)

        time_buf.append(t)
        x_buf.append(x)
        y_buf.append(y)
        z_buf.append(z)

        update_plot()
        time.sleep(REFRESH_RATE)

    except KeyboardInterrupt:
        print("Exiting...")
        break
    except Exception as e:
        print("Error:", e)

ser.close()
