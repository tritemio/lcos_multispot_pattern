#! python3
import socketserver
import yaml
from patternlib import compute_pattern


def process_recv_data(data):
    #print(data)
    params = yaml.load(data)
    print(params, end='\n\n')
    a = compute_pattern(**params)
    return a.tobytes()


class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("- Data received.")
        response = process_recv_data(self.data)
        self.request.sendall(response)

def main():
    HOST, PORT = "localhost", 9999
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
    print('Serving...')
    server.serve_forever()

if __name__ == "__main__":
    main()