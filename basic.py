import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    print("Attempting to connect to the server...")
    client.connect(('192.168.137.1', 5000))  # Replace with the server's IP
    print("Connection successful!")
except socket.timeout:
    print("Connection timed out.")
except socket.error as e:
    print(f"Connection failed: {e}")
finally:
    client.close()
