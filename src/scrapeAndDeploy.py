from sentence_transformers import SentenceTransformer
import torch
from piazza_api import Piazza
import time
import pickle
import tqdm as tqdm
import re

from typing import List, Dict

# Custom class to generate and execute readable SQL queries 
from PostGresQueryGenerator import PGQuery as PGQ

''' 
###### DB Design #######
"Embeddings" Table (
  id              SERIAL          PRIMARY KEY, 
  embedding       vector(768)     NOT NULL, 
  semester_id     INT             NOT NULL, 
  post_id         INT             NOT NULL,
  FOREIGN KEY(semester_id) REFERENCES Semesters(semester_id),
  FOREIGN KEY(semester_id, post_id) REFERENCES Posts(semester_id, post_id),
)

"Semesters" Table (
  semester_id             SERIAL          PRIMARY KEY,
  semester_name           VARCHAR(255),
  semester_piazza_code    TEXT            UNIQUE,
 )

"Posts" Table (
  semester_id         INT     NOT NULL,
  post_id             INT     NOT NULL,
  post_title          TEXT,
  post_content        TEXT,
  instructor_answer   TEXT,
  student_answer      TEXT,
  PRIMARY KEY(semester_id, post_id),
)
'''

EMBEDDINGS_COLUMN = [
                    'id SERIAL PRIMARY KEY', 
                    'embedding vector(768) NOT NULL', 
                    'semester_id INT NOT NULL', 
                    'post_id INT NOT NULL',
                    'FOREIGN KEY(semester_id) REFERENCES Semesters(semester_id)',
                    'FOREIGN KEY(semester_id, post_id) REFERENCES Posts(semester_id, post_id)'
                    ]

SEMESTERS_COLUMN = [
                    'semester_id SERIAL PRIMARY KEY', 
                    'semester_name VARCHAR(255)', 
                    'semester_piazza_code TEXT UNIQUE'
                    ]

POSTS_COLUMN = [
                'semester_id INT NOT NULL', 
                'post_id INT NOT NULL', 
                'post_title TEXT', 
                'post_content TEXT',
                'instructor_answer TEXT', 
                'student_answer TEXT',
                'PRIMARY KEY(semester_id, post_id)'
                ]

TABLES = {
        'Embeddings'  : EMBEDDINGS_COLUMN, 
        'Semesters'   : SEMESTERS_COLUMN, 
        'Posts'       : POSTS_COLUMN
        }

DATABASE_NAME = 'piazzapostdata'

PG_LOGIN = {
    'dbname'    : 'postgres',
    'user'      : 'kannah',
    'password'  : 'macbook',
    'host'      : 'localhost',
    'port'      : '5432',
}

RUN_DATABASE_INTIALIZATION = False
CREATE_NEW_TABLES = False or RUN_DATABASE_INTIALIZATION

EMAIL = ''
PASSWORD = ''

def piazzaLogIn() -> Piazza:
    piazza_obj = Piazza()
    piazza_obj.user_login(email=EMAIL, password=PASSWORD) 
    return piazza_obj

def get360Classes(piazza_obj):
    csci360nid = []
    for i in piazza_obj.get_user_classes():
        if '360' in i['num']:
            csci360nid.append((i['nid'], i['num']))
    return csci360nid

if __name__ == "__main__":

    encoder = SentenceTransformer('bert-base-nli-mean-tokens')
    if RUN_DATABASE_INTIALIZATION:
        print("Initializing db")
        SQL = PGQ()
        SQL.login(PG_LOGIN)
        SQL.toggleAutoCommit()

        ''' 
            1. Drop Database
            2. Create Database
        '''
        SQL.DROP_DATABASE().IF_EXISTS(DATABASE_NAME).execute_nofetch()
        SQL.CREATE_DATABASE(DATABASE_NAME).execute_nofetch()
        SQL.commit()
        SQL.toggleAutoCommit()
    
    '''
        1. Switch to the new database created
        2. Add the vector extension to the database if intialization is required
    '''
    PG_LOGIN['dbname'] = DATABASE_NAME
    SQL = PGQ()
    SQL.login(PG_LOGIN)
    SQL.toggleAutoCommit()
    print("Logging in")
    if RUN_DATABASE_INTIALIZATION:
        SQL.CREATE_EXTENSTION('vector').execute_nofetch()
        SQL.commit()
    
    if CREATE_NEW_TABLES:
        '''
            0. Drop Tables if they exist
            1. Create Semesters Table
            2. Create Posts Table
            3. Create Embeddings Table
        '''
        for table in TABLES.keys():
            SQL.DROP_TABLE().IF_EXISTS(table).execute_nofetch()

        '''
        "Semesters" Table (
        semester_id             SERIAL          PRIMARY KEY,
        semester_name           VARCHAR(255),
        semester_piazza_code    TEXT,
        )
        '''
        SQL.CREATE_TABLE('Semesters', SEMESTERS_COLUMN).execute_nofetch()

        '''
        "Posts" Table (
        semester_id         INT     NOT NULL,
        post_id             INT     NOT NULL,
        post_title          TEXT,
        post_content        TEXT,
        instructor_answer   TEXT,
        student_answer      TEXT,
        PRIMARY KEY(semester_id, post_id),
        )
        '''
        SQL.CREATE_TABLE('Posts', POSTS_COLUMN).execute_nofetch()

        '''
        "Embeddings" Table (
        id              SERIAL          PRIMARY KEY, 
        embedding       vector(768)     NOT NULL, 
        semester_id     INT             NOT NULL, 
        post_id         INT             NOT NULL,
        FOREIGN KEY(semester_id) REFERENCES Semesters(semester_id),
        FOREIGN KEY(semester_id, post_id) REFERENCES Posts(semester_id, post_id),
        )
        '''
        SQL.CREATE_TABLE('Embeddings', EMBEDDINGS_COLUMN).execute_nofetch()
        
        SQL.commit()
        print("Created new tables")

    '''
        1. Load data directly from Piazza
        2. Clean the data
        3. Encode the data
        4. Insert the data into the database
    '''
    FILTER_HTML_TAGS = r'<[^>]*>'
    piazza_obj = piazzaLogIn()
    classNidArr = get360Classes(piazza_obj) # Getting all of the nid's for 360
    for semester_nid, semester_class_name in classNidArr:
        
        # Checking if the semester already exists in the database
        # Avoid overwriting if that is the case
        exists = SQL.SELECT(
                    ['*']
                    ).FROM(
                        ['Semesters']
                    ).WHERE(
                        [f'semester_piazza_code = {PGQ.toString(semester_nid)}']
                    ).execute_fetch()
    
        if len(exists) != 0:
            print("Semester data already acquired")
            continue

        # Loading the semester infromation into the database
        class_network = piazza_obj.network(semester_nid)

        SQL.INSERT_INTO(
            'Semesters', 
            ('semester_name', 'semester_piazza_code')
            ).VALUES([
                    (PGQ.toString(semester_class_name + semester_nid), PGQ.toString(semester_nid))
            ]).execute_nofetch()
        
        SQL.commit()
        # Getting Semester ID from the database
        semester_id = SQL.SELECT(
            ['semester_id']
            ).FROM(
                ['Semesters']
            ).WHERE(
                [f'semester_piazza_code = {PGQ.toString(semester_nid)}']
            ).execute_fetch()[0][0]    
        
        ######### Iterate through all of the posts in the semester and add them to the database #########
        minn = 5
        maxx = 1000
        myClass = piazza_obj.network(semester_nid)
        for post_num in tqdm.tqdm(range(minn, maxx)):
            try:

                ######## Post Filtering ########
                post = myClass.get_post(post_num)    
                if 'regrade' in post['folders'] or post['status'] == 'private':
                    continue

                title = ''
                content = ''
                instructor_answer = ''
                student_answer = ''

                if 'history' in post and post['history']:
                    if 'subject' in post['history'][0]:
                        title = post['history'][0]['subject']
                    
                    if 'content' in post['history'][0]:
                        content = post['history'][0]['content']
                
                if 'children' in post:
                    for child_post in post['children']:
                        if 'type' in child_post and child_post['type'] == 'i_answer':
                            instructor_answer = ' ' + child_post['history'][0]['content']
                            if len(student_answer) > 0:
                                break

                        elif child_post['type'] == 's_answer' and child_post['is_tag_endorse']:
                            student_answer = child_post['history'][0]['content']
                            if len(instructor_answer) > 0:
                                break

                ######## DATABASE INSERTION ########
                # Remvoing HTML tags and trailing whitespace from the data
                cleaned_question = re.sub(FILTER_HTML_TAGS, '', title.strip())
                cleaned_question_content = re.sub(FILTER_HTML_TAGS, '', content.strip())
                cleaned_instructor_answer = re.sub(FILTER_HTML_TAGS, '', instructor_answer.strip())
                cleaned_student_answer = re.sub(FILTER_HTML_TAGS, '', student_answer.strip())

                # Encoding the cleaned data
                encoded_question = encoder.encode(cleaned_question)
                sentence_encoded_content = [encoder.encode(sentence) for sentence in cleaned_question_content.split('.') if len(sentence) > 20]
                sentence_encoded_instructor_answer = [encoder.encode(sentence) for sentence in cleaned_instructor_answer.split('.') if len(sentence) > 20]
                sentence_encoded_student_answer = [encoder.encode(sentence) for sentence in cleaned_student_answer.split('.') if len(sentence) > 20]

                # Making an entry containing entire post information
                SQL.INSERT_INTO('Posts', ('semester_id', 'post_id', 'post_title', 'post_content', 'instructor_answer', 'student_answer'))\
                    .VALUES([(str(semester_id), PGQ.toInt(post_num), PGQ.toString(cleaned_question), PGQ.toString(cleaned_question_content), 
                            PGQ.toString(cleaned_instructor_answer), PGQ.toString(cleaned_student_answer))])\
                        .execute_nofetch()
                
                # Making multiple vector embedding entries for Question + each sentence in the post
                SQL.INSERT_INTO('Embeddings', ('embedding', 'semester_id', 'post_id'))\
                    .VALUES([(PGQ.toVector(encoded_question), PGQ.toInt(semester_id), PGQ.toInt(post_num))])\
                        .execute_nofetch()
                
                for embedding in sentence_encoded_content:
                    SQL.INSERT_INTO('Embeddings', ('embedding', 'semester_id', 'post_id'))\
                    .VALUES([(PGQ.toVector(embedding), PGQ.toInt(semester_id), PGQ.toInt(post_num))])\
                        .execute_nofetch()
                
                for embedding in sentence_encoded_instructor_answer:
                    SQL.INSERT_INTO('Embeddings', ('embedding', 'semester_id', 'post_id'))\
                    .VALUES([(PGQ.toVector(embedding), PGQ.toInt(semester_id), PGQ.toInt(post_num))])\
                        .execute_nofetch()
                
                for embedding in sentence_encoded_student_answer:
                    SQL.INSERT_INTO('Embeddings', ('embedding', 'semester_id', 'post_id'))\
                    .VALUES([(PGQ.toVector(embedding), PGQ.toInt(semester_id), PGQ.toInt(post_num))])\
                        .execute_nofetch()

                SQL.commit()

                ####### Sleeping to avoid getting blocked by Piazza #######
                if post_num%10 == 0:
                    time.sleep(60 * 15) # seconds * minutes
                    piazza_obj = piazzaLogIn()
                    myClass = piazza_obj.network(semester_nid)
                else:
                    time.sleep(60 * 1)
                    myClass = piazza_obj.network(semester_nid)
            except:
                print("Error with post: ", post_num)
                time.sleep(60 * 10)
                myClass = piazza_obj.network(semester_nid)
                continue

        ############################################################
