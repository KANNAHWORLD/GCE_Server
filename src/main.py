import http.server
import socketserver
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PORT = 8000


PING_REQUEST = {
        "message": "Hello from Lambda!"
}
    
# Model and Tokenizers
PRED_MODEL_NAME = "bansalsi/467ArxivClassification"
TOKENIZER_NAME = "bert-base-uncased"
NUM_LABELS = 11

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

        # Loading the model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(PRED_MODEL_NAME, num_labels=NUM_LABELS, force_download=False)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, force_download=False)

        # Toeknize the data
        tokenized_pt_tensor = tokenizer(data, max_length=512, truncation=True, return_tensors="pt")
        
        # get the output from the model 
        outputs = model(**tokenized_pt_tensor)

        # get the prediction from the output of the model
        prediction = np.argmax(outputs.logits.detach().numpy())
        
        # Defining the data to be returned
        data = {"message": LABEL_DESCRIPTIONS[prediction]}

        self.make_good_response(data)


    def do_POST(self):

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        if 'resource' in data:
            if data['resource'] == "arxivClassification" and 'data' in data:
                self.handle_arxiv_classification(data['data'])
                return

        self.make_good_response(PING_REQUEST)



if __name__ == "__main__":

    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", PORT), myHandler)

    print("serving at port", PORT)
    httpd.serve_forever()