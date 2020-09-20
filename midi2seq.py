#######
# Copyright 2020 Jian Zhang, All rights reserved
##
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

from math import ceil
velo_inc = 5
dim = 128*2 + 100 + int(ceil(126/velo_inc))  # This is the size of vocabulary.

import random
import glob
import numpy as np
import pretty_midi

class Event:
    def __init__(self, s, t, v):
        self.time = s
        self.type = t
        self.val = v

    def encode(self):
        if self.type == 'down':
            return self.val
        elif self.type == 'up':
            return 128 + self.val
        elif self.type == 'shift':
            return 128*2 + self.val
        else:
            return 128*2 + 100 + self.val

    @staticmethod
    def decode(code):
        if code < 128:
            return 'down', code
        elif code < 128*2:
            return 'up', code - 128
        elif code < 128*2 + 100:
            return 'shift', (code - 128*2)/100 + 0.01
        else:
            return 'velo', (code - 128*2 - 100)*velo_inc + int(velo_inc/2)


def piano2seq(midi):
    '''
    Convert a midi object to a sequence of events
    :param midi: midi object or the file name of the midi file
    :return: numpy array that contains the sequence of events
    '''
    if type(midi) is str:
        midi = pretty_midi.PrettyMIDI(midi)
    piano = midi.instruments[0]

    velo = 0
    q = []
    for note in piano.notes:
        if note.velocity != velo:
            q.append(Event(note.start, 'velo', int(min(note.velocity, 125)/velo_inc)))
            velo = note.velocity
        q.append(Event(note.start, 'down', note.pitch))
        q.append(Event(note.end, 'up', note.pitch))

    t = 0
    qfull = []
    for e in sorted(q, key=lambda x: x.time):
        d = e.time - t
        while d > 0.01:
            dd = min(d, 1) - 0.01
            qfull.append(Event(t, 'shift', int(dd*100)))
            d = d - dd
        t = e.time
        qfull.append(e)

    seq = np.zeros((len(qfull),), dtype=np.int32)
    for i, e in enumerate(qfull):
        seq[i] = e.encode()

    assert np.max(seq) < dim
    return seq

def seq2piano(seq):
    '''
    Convert a sequence of events to midi
    :param seq: numpy array that contains the sequence
    :return: midi object
    '''
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, is_drum=False, name='piano')
    midi.instruments.append(piano)

    if seq.ndim > 1:
        seq = np.argmax(seq, axis=-1)
    inote = {}
    velo = 40
    time = 0.
    for e in seq:
        t, v = Event.decode(e)
        if t == 'shift':
            time += v
        elif t == 'velo':
            velo = v
            for n in inote.values():
                if n[2] == time:
                    n[0] = v
        elif t == 'down':
            n = inote.get(v, None)
            if n is not None:
                logging.debug('consecutive downs for pitch %d at time %d and %d' % (v, n[2], time))
            else:
                inote[v]  = [velo, v, time, -1]
        else:
            n = inote.get(v, None)
            if n is not None:
                n[-1] = time
                if n[-1] > n[-2]:
                    piano.notes.append(pretty_midi.Note(*n))
                else:
                    logging.debug('note with non-positive duration for pitch %d at time %d' % (n[1], n[2]))
                del inote[v]
            else:
                logging.debug('up without down for pitch %d at time %d' % (v, time))

    # clean out the incomplete note buffer, assuming these note end at last
    for n in inote.values():
        n[-1] = time
        if n[-1] > n[-2]:
            piano.notes.append(pretty_midi.Note(*n))

    return midi

def segment(seq, maxlen=50):
    assert len(seq) > maxlen
    inc = int(maxlen/2)
    i = inc
    t = np.ones((maxlen+1,), dtype=np.int32)
    t[0] = (128*2+1)
    t[1:] = seq[:maxlen]
    s = [t]
    while i+maxlen+1 < len(seq):
        s.append(seq[i:i+maxlen+1])
        i += inc
    return np.stack(s, axis=0)

def process_midi_seq(all_midis=None, datadir='data', n=10000, maxlen=50):
    '''
    Process a list of midis, convert them to sequences and segment sequences into segments of length max_len
    :param all_midis: the list of midis. If None, midis will be loaded from files
    :param datadir: data directory, assume under this directory, we have the "maestro-v1.0.0" midi directory
    :param n: # of segments to return
    :param maxlen: the length of the segments
    :return: numpy array of shape [n', max_len] for the segments. n' tries to be close to n but may not be exactly n.
    '''
    if all_midis is None:
        all_midis = glob.glob(datadir+'/maestro-v1.0.0/**/*.midi')
        random.seed()    # for debug purpose, you can pass a fix number when calling seed()
        random.shuffle(all_midis)

    data = []
    k = 0
    for m in all_midis:
        seq = segment(piano2seq(m), maxlen)
        data.append(seq)
        k += len(seq)
        if k > n:
            break

    return np.vstack(data)

def random_piano(n=100):
    '''
    Generate random piano note
    :param n: # of notes to be generated
    :return: midi object with the notes
    '''
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, is_drum=False, name='piano')
    midi.instruments.append(piano)

    pitchs = np.random.choice(128, size=n)
    velos = np.random.choice(np.arange(10, 80), size=n)
    durations = np.abs(np.random.randn(n) + 1)
    intervs = np.abs(0.2*np.random.randn(n) + 0.3)
    time = 0.5
    for i in range(n):
        piano.notes.append(pretty_midi.Note(velos[i], pitchs[i], time, time+durations[i]))
        time += intervs[i]

    return midi
