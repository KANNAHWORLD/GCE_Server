import http.server
import socketserver
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

PORT = 8000


PING_REQUEST = {
        "message": "Hello from Lambda!"
}

BAD_REQUEST = {
    "message": "Bad Request"
}
    
# Model and Tokenizers
PRED_MODEL_NAME = os.getcwd() + "/ArxivClassificationModel/"
TOKENIZER_NAME = os.getcwd() + "/ArxivClassificationTokenizer/"
NUM_LABELS = 11
model = None
tokenizer = None

# Label and Description Informations
LABELS = ['math.AC', 'cs.CV', 'cs.AI', 'cs.SY', 'math.GR', 'cs.DS', 'cs.CE', 'cs.PL', 'cs.IT', 'cs.NE', 'math.ST']
LABEL_DESCRIPTIONS = [
    "Math: Commutative Algebra",
    "CS: Computer Vision and Pattern Recognition",
    "CS: Artificial Intelligence",
    "CS: Systems and Control",
    "Math: Group Theory",
    "CS: Data Structures and Algorithms",
    "CS: Computer Engineering, Finance, Science",
    "CS: Programming Languages",
    "CS: Information Theory",
    "CS: Neural and Evolutionary Computing",
    "Math: Statistics Theory"
]
        

class myHandler(http.server.SimpleHTTPRequestHandler):

    def make_good_response(self, data):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def handle_arxiv_classification(self, data) -> str:

        global tokenizer
        global model
        tokenized_pt_tensor = tokenizer(data, max_length=512, truncation=True, return_tensors="pt")
        
        # get the output from the model 
        outputs = model(**tokenized_pt_tensor)

        # get the prediction from the output of the model
        prediction = np.argmax(outputs.logits.detach().numpy())
        
        # Defining the data to be returned
        data = {"message": LABEL_DESCRIPTIONS[prediction]}

        print("Classification completed, responding back")

        self.make_good_response(data)


    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            print("Recieved a request")
            if 'resource' in data:
                if data['resource'] == "arxivClassification" and 'data' in data:
                    print("Requested for arxiv classification")
                    self.handle_arxiv_classification(data['data'])
                    return
            print("Recieved a request but not for any endpoint, returning PING_REQUEST")
            self.make_good_response(PING_REQUEST)
        except:
            self.make_good_response(BAD_REQUEST)

    def do_GET(self):
        self.make_good_response(PING_REQUEST)

class CustomHTTPServer(socketserver.TCPServer):
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        # Set a custom socket timeout
        self.socket.settimeout(60)  # Set timeout to 60 seconds


def start_up():
    global model
    global tokenizer

    model = AutoModelForSequenceClassification.from_pretrained(PRED_MODEL_NAME, num_labels=NUM_LABELS, force_download=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, force_download=False)


if __name__ == "__main__":

    # handler = http.server.SimpleHTTPRequestHandler
    # httpd = socketserver.TCPServer(("", PORT), myHandler)
    start_up()
    server = CustomHTTPServer(('localhost', 8080), myHandler)
    print("Serving on port 8080...")
    server.serve_forever()

    # print("serving at port", PORT)
    # httpd.serve_forever()