#!/usr/bin/env python
import subprocess
import datetime
from transcribe_async import call_google
import json
from threading import Thread

THRESHOLD = '8%'
TIME_FORMAT = '%Y%m%d_%H_%M_%S'


def record():
    t = datetime.datetime.now().strftime(TIME_FORMAT)

    filename = 'recording%s.raw' %t
    c = ['rec', '-c', '1', '-r', '16k', '-t', 'raw', filename, 'silence', '0', '1', '00:00:02.0', THRESHOLD]
    subprocess.call(c)

    return filename

def convert(audiofile):
    f = file("transcript_"+audiofile[:-4]+".txt", "w")

    response = call_google(audiofile)
    confidence = float( json.dumps( response['response']['results'][0]['alternatives'][0]['confidence'] ) )
    transcript = json.dumps( response['response']['results'][0]['alternatives'][0]['transcript'] )

    if confidence > 0.5:
        f.write(transcript)
    else:
        f.write("[POOR RECOGNITION]")


if __name__ == "__main__":
    while(True):
        t = Thread( target=convert, args=(record(), ) )
        t.start()
