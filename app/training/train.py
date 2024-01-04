from evaluator import Custom_CEBinaryClassificationEvaluator
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator, CEBinaryClassificationEvaluator, \
    CECorrelationEvaluator
from CrossEncoder import Custom_CrossEncoder
import torch
import torch.nn as nn
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm


def give_data_loader(df, size_=10000, batch_size_=32, num_workers_=0, shuffle_=True):
    examples = []
    for i in tqdm(range(len(df[:size_]))):
        try:
            s1 = df['Query'].iloc[i]
            s2 = df['Document'].iloc[i]
            label = df['label'].iloc[i]
        except:
            s1 = df['sent1'].iloc[i]
            s2 = df['sent2'].iloc[i]
            label = df['label'].iloc[i]

        examples.append(InputExample(texts=[s1, s2], label=label))
    dataloader = DataLoader(examples, batch_size=batch_size_, num_workers=num_workers_, shuffle=shuffle_)
    return dataloader, examples


def train(model, train_dataloader, val_dataloader, train_evaluator, val_evaluator,
          save_model_path='/content/drive/MyDrive/Javis/Entity_Search_Model/reranking_models/cross_encoder_models/experiment_javis_data/miniLM_L6',
          num_epochs=20,
          batch_size_=64,
          base_model_='miniLM_L6',
          evaluation_steps_=1000,
          activation_fct_=nn.Identity(),
          scheduler_='WarmupLinear',
          loss_=nn.BCEWithLogitsLoss(),
          use_wandb_=False,
          wandb_run_name='New Run'):
    # activation_fct_ = nn.Sigmoid()
    # loss_ = nn.CrossEntropyLoss()

    model.fit(train_dataloader=train_dataloader,
              validation_dataloader=val_dataloader,
              train_evaluator=train_evaluator,
              validation_evaluator=val_evaluator,
              activation_fct=activation_fct_,
              loss_fct=loss_,
              epochs=num_epochs,
              evaluation_steps=evaluation_steps_,
              scheduler=scheduler_,
              warmup_steps=int(len(train_dataloader) * num_epochs * 0.01),
              # warmup_steps= 0,
              output_path=save_model_path,
              batch_size=batch_size_,
              dataset_size=len(train_sample),
              base_model=base_model_,
              loss_name=str(loss_),
              use_wandb=use_wandb_,
              optimizer_params={'lr': 2e-5},
              optimizer_class=torch.optim.AdamW,
              wandb_run_name=wandb_run_name
              )
    return model


if __name__ == '__main__':
    # Hyper parameters
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    train_data_length = 180000
    val_data_length = 10000
    batch_size = 64
    epochs = 8
    activation_function = nn.Identity()
    # activation_function =  nn.Sigmoid()
    # loss = nn.CrossEntropyLoss()
    loss = nn.BCEWithLogitsLoss()
    use_wandb = True
    base_model_name = 'miniLM_L6'
    scheduler = 'warmuplinear'  # 'warmuplinear' ,   'constantlr' , 'warmupcosine' , 'warmupcosinewithhardrestarts'  , 'warmupconstant'

    print('----- Data Loading -------')
    HOME_DIR = '/home/ubuntu'
    file_name = 'df_prod_phonetic_mix_v2.csv'
    df = pd.read_csv(HOME_DIR + '/data/' + file_name).sample(frac=1, random_state=42).dropna().drop_duplicates()
    print(f'---Length of Data = {len(df)}----')
    train_data = df[:train_data_length]
    val_data = df[-val_data_length:]

    final_path = f'miniLM_L6_{file_name[:-4]}_{int(train_data_length // 1000)}k'
    save_model_path = f'/home/ubuntu/checkpoints/models_checkpoints/entity_search/reranking_models/' + f"crossencoder_models/" + final_path
    train_dataloader, train_sample = give_data_loader(train_data, batch_size_=batch_size, size_=train_data_length)
    val_dataloader, val_sample = give_data_loader(val_data, batch_size_=batch_size, size_=val_data_length)

    train_evaluator = Custom_CEBinaryClassificationEvaluator.from_input_examples(
        train_sample)  # labels = discrete (0,1)
    val_evaluator = Custom_CEBinaryClassificationEvaluator.from_input_examples(val_sample)
    # train_evaluator = CECorrelationEvaluator.from_input_examples(train_sample)        # labels = continous
    # val_evaluator = CECorrelationEvaluator.from_input_examples(val_sample)

    print('----- Model Loading -------')
    model = Custom_CrossEncoder(MODEL_NAME, num_labels=1,
                                default_activation_function=nn.Sigmoid())

    evaluation_steps = int(0.25 * len(train_dataloader))

    print('----Model Training Started -----')
    model = train(model, train_dataloader, val_dataloader, train_evaluator, val_evaluator,
                  save_model_path=save_model_path, num_epochs=epochs, batch_size_=batch_size,
                  evaluation_steps_=evaluation_steps,
                  scheduler_=scheduler,
                  base_model_=base_model_name, activation_fct_=activation_function, loss_=loss, use_wandb_=use_wandb,
                  wandb_run_name=final_path)
