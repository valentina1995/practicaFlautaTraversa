import librosa
import pandas as pd
import numpy as np
import pretty_midi


# -----------------Functions for uploading audio and MIDI files-------

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'midi', 'mid'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------App features--------------------------


def extractHopLenght(y, sr):
  bmp = librosa.beat.tempo(y, sr)
  bmp = bmp[0]
  if (0 <= bmp or bmp <= 90 ):
    hop_length = 512
  elif (90 < bmp ):
    hop_length = 256
  return hop_length, bmp

def divSignal(signal, hop_length):
  """ Returns:
        sep : list
        Contains the signal fragments where there is no silence separated in each array
        senaldiv : list
        Contains the beginnings and endings in samples of the intervals that do not have silences"""
  senaldiv = librosa.effects.split(signal,top_db = 15, hop_length=hop_length )
  sep = []
  for i in range(len(senaldiv)):
     
      x = signal[slice(senaldiv[i,0],senaldiv[i,1])]
      sep.append(x)
  return sep,senaldiv

def estimate_pitch(segment, sr, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C6')):
    
    # Compute autocorrelation of input segment.

    r = librosa.autocorrelate(segment)
    
    # Define lower and upper limits for the autocorrelation argmax.
    i_min = sr/fmax
    i_max = sr/fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0
    
    # Find the location of the maximum autocorrelation.
    i = r.argmax()
    f0 = float(sr)/i
    return f0

def estimate_pitch_fft(segment, sr):
  # Compute a fft of input segment.
  L = len(segment)
  w = int(np.floor(L/2))-1 
  frec = np.linspace (0,1,L)*sr
  S=np.abs(np.fft.fft(segment)/L)[:w]
   #Choose the peak
  freq = frec[S.argmax()]
  return freq

def get_df_audio(sep, div, bmp, sr):
    audio_list = []
    for i in range(0,len(sep)):
         duration = librosa.core.get_duration(sep[i])
         start = librosa.samples_to_time(div[i,0])
         end = librosa.samples_to_time(div[i,1])
         pitch = estimate_pitch_fft(sep[i],sr)
         #pitch = estimate_pitch(sep[i], sr)
         midi = librosa.hz_to_midi(pitch)
         audio_list.append([start, end, duration, pitch, midi, bmp])
    df = pd.DataFrame(audio_list, columns=['start', 'end', 'duration', 'pitch','midi', 'tempo'])    
    return df

def get_df_mid(midi_data):
    midi_list = []
    """Temporary change where you choose the lowest value  """
    tempo = midi_data.estimate_tempi()
    pos = tempo[0]
    cal = tempo[1]
    tempo = pos[cal.argmin()]
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            midi = note.pitch
            pitch = pretty_midi.note_number_to_hz(midi)
            duration = pretty_midi.Note.get_duration(note)
            midi_list.append([start, end, duration, midi, pitch, tempo])
    mid_df = pd.DataFrame(midi_list, columns=[
                          'start', 'end', 'duration', 'midi', 'pitch', 'tempo'])
    return mid_df
  
def note_calification(df_audio, df_mid):
    df_audio =df_audio.round(4)
    df_mid = df_mid.round(4)
    """The divisor is always the reference"""
    result = []
    for i in range (0, len(df_audio)):
        delta = np.log2(df_audio.pitch[i]/ df_mid.pitch[i]) * 1200
        print(delta)
        if ((delta >= 1) and (delta <= 50)):
                nota = 0.5
        elif ((delta <= -1) and (delta >= -50)):
          nota = -0.5
        elif ((delta > -1) and (delta < 1)):
          nota = 1
        else :
          nota = 0

        if (df_audio.tempo[i] == df_mid.tempo[i]):
          dif = (df_audio.duration[i]-abs(df_audio.duration[i]-df_mid.duration[i])) / df_audio.duration[i]
        else:
          #I determine the note that should have been played
          #note : 'String' note name
          nota_base = delay_note(df_mid.tempo[i],df_mid.duration[i])
          #Exercise dictionary
          dic_ejx = sec_to_note(df_audio.tempo[i])
          ideal = dic_ejx.get(nota_base)
          dif = (ideal -abs(df_audio.duration[i]- ideal)) / ideal
        if dif <= 0.20:
          nota_tempo = 1
        elif (dif > 0.20) & (dif < 0.50):
          nota_tempo = 0.5
        else: 
          nota_tempo = 0
        result.append([librosa.hz_to_note(df_mid.pitch[i]),nota, nota_base, nota_tempo])

    r = pd.DataFrame(result, columns=['note','intonation', 'figure', 'duration'])
    wr_note = np.sum(r.intonation !=1)
    wr_time = np.sum(r.duration!=1)
    intonation = r.intonation.abs()
    overall = intonation.mean()+r.duration.mean() / 2
    overall = np.round(overall,2)*100
    return r , wr_note, wr_time , overall
                         

def sec_to_note(bmp):
    quarter = round(60/bmp, 4)
    half = round(quarter * 2, 4)
    whole = round(quarter * 4, 4)
    eighth = round(quarter / 2, 4)
    sixteenth = round(quarter / 4, 4)
    dotted_half = round(quarter * 3 , 4)
    dotted_quarter = round(quarter * 1.5, 4)
    notes = {'Crotchet': quarter, 'Minim': half, 'Semibreve': whole, 'Quaver': eighth,
             'Semiquaver': sixteenth, 'D0tted crotchet': dotted_quarter, 'Dotted minim': dotted_half}
    return notes

def delay_note(bmp, note):
    notes = sec_to_note(bmp)
    for key, value in notes.items():
        if note == value:
            return key
    return 'Cannot qualify'


# ---------------------------------Utils---------------------------------------------
def nextPowerOf2(n):
    p = 1
    if (n and not(n & (n - 1))):
        return n
    while (p < n):
        p <<= 1
    return p


def pad(A, length):
    arr = np.zeros(length)
    arr[:len(A)] = A
    return arr
