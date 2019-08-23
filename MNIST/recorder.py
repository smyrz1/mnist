import logging

log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

KEYSPACE = "mnist_key"


def createKeySpace():
    cluster = Cluster(contact_points=['mnist_cassandra'], port=9042)
    session = cluster.connect()
    log.info("Creating keyspace...")
    try:
        session.execute("""
            CREATE KEYSPACE %s
            WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '3' }
            """ % KEYSPACE)

        log.info("Setting keyspace...")
        session.set_keyspace(KEYSPACE)

        log.info("Creating table...")
        session.execute("""
            CREATE TABLE record_data(
                date timestamp,
                result text,
                PRIMARY KEY (date)
            )
            """)
    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)


def insertData(date, result):
    cluster = Cluster(contact_points=['mnist_cassandra1'], port=9042)
    session = cluster.connect(KEYSPACE)
    log.info("Inserting data...")
    try:
        session.execute(""" 
            INSERT INTO record_data (date, result)
            VALUES(%s, %s);
            """, (date, result))
    except Exception as e:
        log.error("Unable to insert data")
        log.error(e)
