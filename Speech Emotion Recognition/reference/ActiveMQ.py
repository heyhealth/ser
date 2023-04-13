# Send a Message to an Apache ActiveMQ Queue
import stomp

conn = stomp.Connection10()

conn.start()

conn.connect()

conn.send('SampleQueue', 'Simples Assim')

conn.disconnect()

# Receive a Message from an Apache ActiveMQ Queue
import stomp
import time


class SampleListener(object):
    def on_message(self, headers, msg):
        print(msg)


conn = stomp.Connection10()

conn.set_listener('SampleListener', SampleListener())

conn.start()

conn.connect()

conn.subscribe('SampleQueue')

time.sleep(1)  # secs

conn.disconnect()

# Send a Message to an Apache ActiveMQ Topic
import stomp

conn = stomp.Connection10()

conn.start()

conn.connect()

conn.send('/topic/SampleTopic', 'Simples Assim')

conn.disconnect()

# Receive a Message from an Apache ActiveMQ Topic （1）
import stomp
import time


class SampleListener(object):
    def on_message(self, headers, msg):
        print(msg)


conn = stomp.Connection10()

conn.set_listener('SampleListener', SampleListener())

conn.start()

conn.connect()

conn.subscribe('/topic/SampleTopic')

time.sleep(1)  # secs

conn.disconnect()

# Receive a Message from an Apache ActiveMQ Topic （2）
import stomp
import time


class SampleListener(object):
    def on_message(self, headers, msg):
        print(msg)


conn = stomp.Connection10()

conn.set_listener('SampleListener', SampleListener())

conn.start()

conn.connect(headers={'client-id': 'SampleClient'})

conn.subscribe(destination='/topic/SampleTopic', headers={'activemq.subscriptionName': 'SampleSubscription'})

time.sleep(1)  # secs

conn.disconnect()