#!/usr/bin/env python
import subprocess
import datetime
from googleAPICaller import call_google
import json
from threading import Thread

THRESHOLD = '8%'
TIME_FORMAT = '%Y%m%d_%H_%M_%S'
FILE_FORMAT = 'raw'

def record():
    t = datetime.datetime.now().strftime(TIME_FORMAT)

    filename = 'recording%s.%s' %(t,FILE_FORMAT)
    c = ['rec', '-c', '1', '-r', '16k', '-t', FILE_FORMAT, filename, 'silence', '0', '1', '00:00:02.0', THRESHOLD]
    subprocess.call(c)

    return filename

def getTranscript(audiofile):
    f = file(audiofile[:-4]+".txt", "w")

    what = convert(audiofile)
    who = whoIsThis(audiofile)

    f.write(who +" : "+what+"\n")

def convert(audiofile):

    response = call_google(audiofile)
    confidence = float( json.dumps( response['response']['results'][0]['alternatives'][0]['confidence'] ) )
    transcript = json.dumps( response['response']['results'][0]['alternatives'][0]['transcript'] )

    if confidence > 0.4:
        return transcript
    else:
        return "POOR RECOGNITION"

def whoIsThis(audiofile):

    filename = audiofile[:-4]+".wav"

    c = ('sox -r 16k --bits 16 -e signed-integer -c 1 -t raw %s %s' % (audiofile, filename)).split()
    subprocess.call(c)

    c = ('./speaker-recognition/src/speaker-recognition.py -t predict -i %s -m ./speaker-recognition/src/model.out' % filename).split()
    s = subprocess.check_output(c).strip()

    who = s[s.index('-> ')+3:].strip()
    return who

if __name__ == "__main__":
    while(True):
        t = Thread( target=getTranscript, args=(record(), ) )
        t.start()
