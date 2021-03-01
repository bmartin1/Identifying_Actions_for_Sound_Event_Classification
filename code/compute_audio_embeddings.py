import pandas as pd
import numpy as np
import soundfile as sf
import openl3
import os

def compute_audio_embeddings(audio_path, metadata):

    metadata_file = pd.read_csv(metadata)
    filenames = metadata_file['filename'].tolist()
    audio_list = []
    
    # Best performing parameters
    content_type = "music" 
    input_repr="mel256" 
    embedding_size=6144

    for filename in filenames:
        audio, sr = sf.read(audio_path + filename)
        audio_list.append(audio)

    
    model = openl3.models.load_audio_embedding_model(content_type=content_type,
                                   input_repr=input_repr, embedding_size=embedding_size)

    emb_list, _ = openl3.get_audio_embedding(audio_list, sr, model=model) 


    ##Save the 2,000 filenames-embeddings into a numpy array
    dataset = tuple(zip(filenames,emb_list))
    np.save('ESC-50_openl3_'+content_type+'_'+input_repr+'_'+str(embedding_size)+'.npy', dataset)
    print('Features computed')
    

##Once dowloaded ESC-5- dataset, update both directories and call the function
metadata = 'ESC-50-master/meta/esc50.csv'
audio_path = 'ESC-50-master/audio/'
compute_audio_embeddings(audio_path, metadata)
