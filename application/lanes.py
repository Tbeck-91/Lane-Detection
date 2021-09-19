"""
FILE					: detect_lanes.py  
PROJECT				: Aceis Group Inc.  
AUTHOR				: Taylor Beck 
CO-AUTHORS    : Travis Roy & Felicia Neeb
FIRST VERSION	: 2021-09-16
DESCRIPTION		: Contains message queue logic
"""
# Imports
import pika
import os
import time
import json

from run_detections import run_detection

hostname = os.getenv('RABBITMQ_HOSTNAME')
username = os.getenv('RABBITMQ_DEFAULT_USER')
password = os.getenv('RABBITMQ_DEFAULT_PASS')
queue = os.getenv('RABBITMQ_CONSUME_QUEUE')

class detectLanes(object):
  """
  Get Figure
  This uses a connection to the back-end using RabbitMQ to process a road image and detect lanes

  Parameters:
    object: base 64 encoded image - needs to be a 3 channel image (jpeg)

  Returns: The figure number for the worksite
  """
  def __init__(self):
    """Starting the RabbitMQ Server to read messages from the APIQueue
    
    Note: Once the username and password is changed you will need to update it here."""
    
    # Ensuring all environment variables are set correctly
    if all(v is not None for v in [hostname, username, password]):
      rabbitMQReady = False
      while rabbitMQReady == False:
        try:
          credentials = pika.PlainCredentials(username, password)
          self.connection = pika.BlockingConnection(pika.ConnectionParameters(hostname, 5672, '/', credentials ))
          self.channel = self.connection.channel()
          self.channel.basic_consume(queue=queue, on_message_callback=self.on_response, auto_ack=True)
          rabbitMQReady = True
          print("Connected to RabbitMQ server!")
        except:
          print("Unable to connect to RabbitMQ server")
          print("Attempting to connect in 20s...")
          time.sleep(20)

      self.channel.start_consuming()
    else:
      raise ValueError('Environment variables not set correctly!')

  def on_response(self, ch, method, props, body):
    """Sets response as the body of the message queue.
      Parses the message.
    
    Parameters:
      body: The body of the message form the message queue"""
    self.response = body
    self.props = props
    self.parse_message(body)

  def call(self, payload):
    """Sends messages to the Queue
    
    Parameters: 
      payload: The message to be sent to the Queue"""
    payloadAsString = json.dumps(payload)
    self.response = None
    self.channel.basic_publish(exchange='', routing_key=self.props.reply_to, properties=pika.BasicProperties(correlation_id = \
                                                         self.props.correlation_id), body=payloadAsString)
    while self.response is not None:
      self.connection.process_data_events()

  def parse_message(self, body):
    """Parses the message from the Message queue to ...
    
    Parameters: 
      body: The message from the API to parse"""
    print("Parsing API message")
    
    if not body:
      return {"message": "An internal error occurred"}
    
    image = json.loads(body)
    detection_polys = run_detection(image)
    self.call(detection_polys)

    


  