import os
import json
import numpy as np
import http.server
import socketserver
from PostGresQueryGenerator import PGQuery as PGQ
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

########## SERVING CONFIGURATIONS ##########
PORT = 8000
DEFAULT_IP = '0.0.0.0'


########## DEFAULT RESPONSES ##########
PING_REQUEST = {
        "message": "Hello from Lambda!"
}
BAD_REQUEST = {
    "message": "Bad Request"
}

################# MODEL AND TOKENIZER #################
################# ARXIV CLASSIFICATION ################    
arxiv_classif_model = None
arxiv_classif_tokenizer = None

NUM_LABELS = 11
PRED_MODEL_NAME = os.getcwd() + "/ArxivClassificationModel/"
TOKENIZER_NAME = os.getcwd() + "/ArxivClassificationTokenizer/"


################ LABELS AND DESCRIPTIONS ################
################ ARXIV CLASSIFICATION ###################
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


############## TOKENIZER AND SQL CONNECTION ##############
############## 360 PIAZZA DATABASE #######################
piazza_db_tokenizer = None
piazza_db_connection = PGQ()
POSTGRES_LOGIN = {
    'port'      : '5432',
    'user'      : 'kannah',
    'password'  : 'macbook',
    'host'      : 'localhost',
    'dbname'    : 'piazzapostdata',
}

########## CUSTOM HANDLER ##########
class SidHubHttpServer(http.server.SimpleHTTPRequestHandler):

    def make_good_response(self, data):
        """
        Sends a good response with the given data.

        Parameters:
        - data: JSON format, the data to be sent in the response.

        Returns:
        None
        """
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def handle_arxiv_classification(self, data: str) -> None:
        """
        Handles the classification of arXiv data.
        Writes directly to http response upon completion.
        Args:
            data: The input data to be classified.
        Returns:
            None
        Raises:
            None
        """
        global arxiv_classif_tokenizer
        global arxiv_classif_model
        tokenized_pt_tensor = arxiv_classif_tokenizer(data, max_length=512, truncation=True, return_tensors="pt")
        
        # get the output from the model 
        outputs = arxiv_classif_model(**tokenized_pt_tensor)

        # get the prediction from the output of the model
        prediction = np.argmax(outputs.logits.detach().numpy())
        
        # Defining the data to be returned
        data = {"message": LABEL_DESCRIPTIONS[prediction]}
        print("Classification completed, responding back")
        self.make_good_response(data)

    def handle_360_Piazza_Database(query: str) -> None:
        """
        Handles the 360PiazzaDatabase request.
        Writes directly to http response upon completion.
        Args:
            data: The input data to be processed.
        Returns:
            None
        Raises:
            None
        """
        global piazza_db_tokenizer
        global piazza_db_connection
        
        # Get the embeddings for the data
        query_embedding = piazza_db_tokenizer.encode(query)

        db_response = piazza_db_connection.SELECT([
            "s.semester_name", 
            "s.semester_id",
            "e.post_id", 
            "p.post_title", 
            "p.post_content", 
            "p.instructor_answer", 
            "p.student_answer",
        ]).FROM([
            'embeddings AS e'
        ]).LEFT_JOIN(
            'posts AS p'
        ).ON(
            'p.post_id = e.post_id'
        ).AND(
            'p.semester_id = e.semester_id'
        ).LEFT_JOIN(
            'semesters AS s'
        ).ON(
            's.semester_id = e.semester_id'
        ).GROUP_BY([
            's.semester_id',
            'e.post_id',
        ]).ORDER_BY([
            f'embedding <=> {PGQ.toVector(query_embedding)} DESC',
        ]).LIMIT(
            10
        ).execute_fetch()

        print(db_response)
        return

        # Defining the data to be returned
        data = {"message": "360PiazzaDatabase"}
        print("360PiazzaDatabase completed, responding back")
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
                elif data['resource'] == '360PiazzaDatabase' and 'data' in data:
                    print("Requested for 360PiazzaDatabase")
                    # self.handle_360PiazzaDatabase(data)
                    return
            print("Recieved a request but not for implemented endpoint, returning PING_REQUEST")
            self.make_good_response(PING_REQUEST)
        except:
            self.make_good_response(BAD_REQUEST)

    def do_GET(self):
        """
        Handles the GET request.
        This method is called when a GET request is received by the server. 
        It processes the request and generates a response.
        Currently it only returns a PING_REQUEST response, available for future implementation.
        Parameters:
        - self: The instance of the WebServer class.
        Returns:
        - None
        """
        self.make_good_response(PING_REQUEST)

    def do_CONNECT(self):
        """
        Handles the CONNECT method.
        This method sends a 405 response with a message indicating that the CONNECT method is not allowed.
        Parameters:
            self (WebServer): The instance of the WebServer class.
        Returns:
            None
        """

        self.send_response(405)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"CONNECT method is not allowed.")


class CustomHTTPServer(socketserver.TCPServer):
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        # Set a custom socket timeout
        self.socket.settimeout(120)  # Set timeout to 60 seconds


def start_up():
    global arxiv_classif_model
    global arxiv_classif_tokenizer
    global piazza_db_tokenizer

    arxiv_classif_model = AutoModelForSequenceClassification.from_pretrained(PRED_MODEL_NAME, num_labels=NUM_LABELS, force_download=False)
    arxiv_classif_model.eval()
    arxiv_classif_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, force_download=False)

    # Pull from Cache if available, else download
    piazza_db_tokenizer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    piazza_db_tokenizer.eval()
    piazza_db_connection.login(POSTGRES_LOGIN)

TEST_FLAG = True

if __name__ == "__main__":

    start_up()

    if TEST_FLAG:
        server = SidHubHttpServer.handle_360_Piazza_Database("What is the meaning of life?")
        exit(0)

    server = CustomHTTPServer((DEFAULT_IP, PORT), SidHubHttpServer)
    print(f"Serving on port {PORT}")
    server.serve_forever()
