from flask import Blueprint, request, jsonify
import petl as etl 
from sqlalchemy import create_engine
import openpyxl
from io import BytesIO
import requests

advance_join = Blueprint('advance_join', __name__)


@advance_join.route('/jj', methods=['GET'])
def join_test():
    if request.method == 'GET':
        return jsonify({'join_test_version': '0.0.0',
                        'join_test_api_version': '1.1.1'})
def get_from_excel(cnx):

        link  = cnx['link']
        sheet  = cnx['sheet']

        file = requests.get(link).content
        
      
        result = {}
        excel_file = openpyxl.load_workbook(filename=BytesIO(file))

        tab = etl.fromxlsx(filename=BytesIO(file),sheet =sheet)
        return tab

def get_from_csv(cnx):

        link  = cnx['link']
        tab = etl.fromcsv(link) 

        return tab


def get_from_mysql(cnx):

        _PORT  = cnx['port']
        _HOST  = cnx['host']
        _USER  = cnx['user']
        _PASS  = cnx['pass']
        _DATABASE  = cnx['database']
        _TABLE  = cnx['table']

        engine = create_engine('mysql+pymysql://'+_USER+':'+_PASS+'@'+_HOST+':'+_PORT)

        engine.connect().execute("USE "+_DATABASE +" ;")

        rows = engine.execute('SELECT * FROM '+_DATABASE+'.'+_TABLE +' LIMIT 100')

        data =[dict(row) for row in rows]

        tab  = etl.fromdicts(data)   
        
        return tab

def mapping(obj):
        type = obj["type"]
        tab  ={}
        if(type =="excel"):
               tab =  get_from_excel(obj["cnx"])
        elif(type == "csv"):
               tab =  get_from_csv(obj["cnx"])
        elif(type == "mysql"):
               tab=  get_from_mysql(obj["cnx"])
        
        if len(obj["header"]) > 0:
               tab  = etl.cat(tab, header=obj["header"])
        return tab


def join_table(type,tab_L,tab_R,key_L,key_R):
        if(type =="join"):
        # equi-join
               return etl.join(tab_L,tab_R,lkey=key_L,rkey=key_R)
        elif(type == "leftjoin"):
        # left outer join
               return etl.leftjoin(tab_L,tab_R,lkey=key_L,rkey=key_R)
        elif(type == "lookupjoin"):
        # left join, but where the key is not unique in the right-hand table, arbitrarily choose the first row and ignore others
               return etl.lookupjoin(tab_L,tab_R,lkey=key_L,rkey=key_R)
        elif(type == "rightjoin"):
        # right outer join 
               return etl.rightjoin(tab_L,tab_R,lkey=key_L,rkey=key_R)
        elif(type == "outerjoin"):
        # full outer join 
               return etl.outerjoin(tab_L,tab_R,lkey=key_L,rkey=key_R)
        elif(type == "crossjoin"):
        # cartesian product 
               return etl.crossjoin(tab_L,tab_R)
        elif(type == "antijoin"):
        #  the left table where the key value does not occur in the right table
               return etl.antijoin(tab_L,tab_R,lkey=key_L,rkey=key_R)
        elif(type == "complement"):
        # rows in tab1 that are not in tab2
               return etl.complement(tab_L,tab_R)
        # rows in a that are also in b
        elif(type == "intersection"):
               return etl.intersection(tab_L,tab_R)
        elif(type == "stack"):
        # Concatenate tables        
               return etl.stack(tab_L,tab_R)
        elif(type == "cat"):
        # Concatenate tables
               return etl.cat(tab_L,tab_R)
        
@advance_join.route('/v1/join', methods=['POST'])
def join():
    if request.method == 'POST':
        
       json_data = request.get_json()

       result = {}

       tab_L = mapping(json_data[0])
       key_L = json_data[0]["key"]
       tab_R = mapping(json_data[1])
       key_R = json_data[1]["key"]
       type_join = json_data[0]["join"]

       tab  = join_table(type_join,tab_L,tab_R,key_L,key_R)

       df = etl.todataframe(tab)

       result["data"] = list(etl.dicts(tab))
       result["categorical"] = []
       result["numerical"] = []
       
       for var in df.columns:

              if df[var].dtypes=='O':
                     result["categorical"].append(var)
              else:
                     if len(df[var].unique())<20:
                            result["categorical"].append(var)
                     else:
                            result["numerical"].append(var)


        # Step 5: Return the response as JSON
       return jsonify(result) 

