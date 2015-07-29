__author__ = "Cameron Palone"
__copyright__ = "Copyright 2015, Cameron Palone"
__credits__ = ["Cameron Palone"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Cameron Palone"
__email__ = "cam@cpalone.me"
__status__ = "Prototype"

import argparse
import json
import logging
import pickle
import time

from websocket import create_connection, WebSocketConnectionClosedException, WebSocketException

import data
import models

class MarkovBot:
    def __init__(self, model_path=None, room="test", password=None):
        if model_path is not None:
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
            except FileNotFoundError:
                pass
            self.model_path = model_path
        else:
            self.model = models.TrigramBackoffLM()
            self.model_path = room + ".pickle"

        self.word_list = data.load_word_list()
        self.msg_id = 0
        self.room = room
        self.password = password
        self.conn = None
        self._connect_and_auth()

    def _connect_and_auth(self):
        self.conn = create_connection("wss://euphoria.io/room/{}/ws".format(self.room))
        if self.password is not None:
            self._auth()

    def _send_packet(self, packet):
        try:
            ret = self.conn.send(json.dumps(packet))
            self.msg_id += 1
            return ret
        # TODO: handle reconnect delays better
        except WebSocketConnectionClosedException:
            time.sleep(3)
            logging.warning("Connection closed. Attempting reconnect.")
            self._connect_and_auth()
            return self._send_packet(packet)

    def _auth(self):
        packet = {"type": "auth",
                  "data": {"type": "passcode",
                           "passcode": self.password},
                  "id": str(self.msg_id)}
        return self._send_packet(packet)

    def _handle_ping_event(self, packet):
        # TODO: spin pruning off into separate process/thread
        reply = {"type": "ping-reply",
                 "data": {"time": packet["data"]["time"]},
                 "id": str(self.msg_id)}
        return self._send_packet(reply)

    def _set_nick(self):
        logging.debug("Sending nick.")
        packet = {"type": "nick",
                  "data": {"name": "MarkovBot"},
                  "id": str(self.msg_id)}
        return self._send_packet(packet)

    def _send_message(self, text, parent):
        logging.debug("Sending message with text: %s", text)
        packet = {"type": "send",
                  "data": {"content": text,
                           "parent": parent},
                  "id": str(self.msg_id)}
        return self._send_packet(packet)

    def _handle_send_event(self, packet):
        logging.debug("Received send-event.")
        if packet["data"]["content"][0] != "!" and packet["data"]["sender"] != "MaiMai":
            self.model.update([packet["data"]["content"]], self.word_list)
        elif packet["data"]["content"].startswith("!generate"):
            logging.info("Generating a sentence...")
            self._send_message(self.model.generate(), packet["data"]["id"])

    def _dispatch(self, packet):
        logging.info("Received packet.")
        if packet["type"] == "ping-event":
            logging.info("Handling ping-event.")
            self._handle_ping_event(packet)
        elif packet["type"] == "send-event":
            logging.info("Handling send-event.")
            self._handle_send_event(packet)

    def run(self):
        self._set_nick()
        while(True):
            if self.msg_id % 10 == 0:
                self.model.save(self.model_path)
            try:
                rawdata = self.conn.recv()
                packet = json.loads(rawdata)
            except WebSocketConnectionClosedException:
                time.sleep(3)
                try:
                    self._connect_and_auth()
                except WebSocketException as e:
                    logging.error(e)
                self._set_nick()
            else:
                self._dispatch(packet)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an instance of MarkovBot.")
    parser.add_argument("-m", type=str, help="path to use for model storage")
    parser.add_argument("-p", type=str, help="optional password for the room")
    parser.add_argument("room", type=str,help="room to run bot in")
    args = parser.parse_args()

    non_none_args = {"room": args.room}
    if args.m is not None:
        non_none_args["model_path"] = args.m
    if args.p is not None:
        non_none_args["password"] = args.p
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Starting up.")
    bot = MarkovBot(**non_none_args)
    bot.run()
    # bot = models.TrigramBackoffLM()
    # with open("data/space_log.json") as f:
    #     log = json.load(f)
    # bot.update(log)
    # bot.save("space_log.pickle")