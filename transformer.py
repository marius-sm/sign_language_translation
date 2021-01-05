from transformers import EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, BertGenerationConfig
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders
import datasets
import time
import os
from .training import train_model
import torch.nn.functional as F

def create_tokenizer(sentence_list):
    filename = f'temp_{time.strftime("%Y%m%d-%H%M%S")}.txt'
    with open(filename, 'w') as f:
        for s in sentences:
            f.write(f'{s}\n')

    tokenizer = Tokenizer(WordPiece())
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.enable_padding(pad_token='[PAD]', pad_id=0)

    trainer = WordPieceTrainer(vocab_size=3000, special_tokens=['[PAD]', '[S]', '[/S]', '[UNK]'])
    tokenizer.train(trainer, [filename])

    os.remove(filename)

    return tokenizer

def create_slt_transformer(input_vocab_size=1, output_vocab_size=1, **bert_params):

    if input_vocab_size == 1:
        print('WARNING: Input vocab size is 1')
    if output_vocab_size == 1:
        print('WARNING: Output vocab size is 1')

    params = {
        'vocab_size': input_vocab_size,
        'hidden_size': 512,
        'intermediate_size': 2048,
        'max_position_embeddings': 500,
        'num_attention_heads': 8,
        'num_hidden_layers': 3,
        'hidden_act': 'relu',
        'type_vocab_size': 1,
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1
    }
    params.update(bert_params)

    config = BertGenerationConfig(**params)
    encoder = BertGenerationEncoder(config=config)

    params['vocab_size'] = output_vocab_size
    decoder_config = BertGenerationConfig(is_decoder=True, add_cross_attention=True, **params)
    decoder = BertGenerationDecoder(config=decoder_config)

    transformer = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f'The encoder has {count_parameters(encoder):,} trainable parameters')
    print(f'The decoder has {count_parameters(decoder):,} trainable parameters')
    print(f'The whole model has {count_parameters(transformer):,} trainable parameters')

    return transformer

def encode_string_list(strings, tokenizer, prepend='', append=''):
    # the pad token should be defined by prealably setting tokenizer.enable_padding(pad_token='[PAD]', pad_id=0)
    strings = [prepend + s + append for s in strings]
    encoded_strings = tokenizer.encode_batch(strings)
    ids = torch.tensor([x.ids for x in encoded_strings], dtype=torch.long)
    attmask = torch.tensor([x.attention_mask for x in encoded_strings], dtype=torch.long)
    return ids, attmask

def make_gloss2text_collate_fn(gloss_tokenizer, text_tokenizer, bos_token='[S]', eos_token='[/S]'):

    def collate_fn(inputs):
        # inputs is expected to be a list of pairs (gloss string, text string)
        gloss, text = zip(*inputs)
        gloss_ids, gloss_attmask = encode_string_list(strings=gloss, tokenizer=gloss_tokenizer)
        text_ids, text_attmask = encode_string_list(strings=text, tokenizer=text_tokenizer, prepend=bos_token, append=eos_token)
        return gloss_ids, gloss_attmask, text_ids, text_attmask

    return collate_fn

def make_gloss2text_training_step_fn():

    def training_step_fn(model, batch):

        device = next(model.parameters()).device

        gloss_ids, gloss_attmask, text_ids, text_attmask = batch
        gloss_ids, gloss_attmask, text_ids, text_attmask = gloss_ids.to(device), gloss_attmask.to(device), text_ids.to(device), text_attmask.to(device)

        outputs = model(
            input_ids=gloss_ids,
            attention_mask=gloss_attmask,
            decoder_input_ids=text_ids, 
            decoder_attention_mask=text_attmask,
            return_dict=True
        )
                            
        logits = outputs.logits
        shifted_logits = logits[:, :-1, :].contiguous()
        targets = text_ids[:, 1:].contiguous()
        vocab_size = logits.shape[-1]
        mask = (targets != 0).float()
        num_samples = mask.sum()
        
        loss = F.cross_entropy(shifted_logits.view(-1, vocab_size), targets.view(-1), reduce=False)
        loss *= mask.flatten()
        summed_loss = loss.sum()
                
        predicted = shifted_logits.argmax(2)
        num_correct_samples = ((predicted == targets).float()*mask).sum()
        
        return num_samples, num_correct_samples, summed_loss

    return training_step_fn

def train_gloss2text(model, trainloader, validloader=None, epochs=2, optimizer=None, print_every=1, on_epoch_end=lambda x: x):
    training_step_fn = make_gloss2text_training_step_fn()
    return train_model(
        model,
        training_step_fn=training_step_fn,
        trainloader=trainloader,
        validloader=validloader,
        epochs=epochs,
        optimizer=optimizer,
        print_every=print_every,
        on_epoch_end=on_epoch_end
    )

def make_embeddings2text_collate_fn(text_tokenizer, bos_token='[S]', eos_token='[/S]'):

    def collate_fn(inputs):
        # inputs is expected to be a list of pairs (embeddings, sentence)
        # where embeddings is a tensor of shape (length, embedding dim)
        # and sentence is a string

        embeddings = [i[0] for i in inputs]
        sentences = [i[1] for i in inputs]

        max_length = max([e.shape[0] for e in embeddings])

        inputs_embeds = torch.zeros(len(inputs), max_length, embeddings[0].shape[-1], dtype=torch.float32)
        attmask = torch.zeros(len(inputs), max_length, dtype=torch.long)
        for i, e in enumerate(embeddings):
            inputs_embeds[i, :e.shape[0], :] = e
            attmask[i, :e.shape[0]] = 1

        text_ids, text_attmask = encode_string_list(strings=sentences, tokenizer=text_tokenizer, prepend=bos_token, append=eos_token)

        return inputs_embeds, attmask, text_ids, text_attmask

    return collate_fn

def make_embeddings2text_training_step_fn():

    def training_step_fn(model, batch):

        device = next(model.parameters()).device

        inputs_embeds, attmask, text_ids, text_attmask = batch
        inputs_embeds, attmask, text_ids, text_attmask = inputs_embeds.to(device), attmask.to(device), text_ids.to(device), text_attmask.to(device)

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attmask,
            decoder_input_ids=text_ids, 
            decoder_attention_mask=text_attmask,
            return_dict=True
        )
                            
        logits = outputs.logits
        shifted_logits = logits[:, :-1, :].contiguous()
        targets = text_ids[:, 1:].contiguous()
        vocab_size = logits.shape[-1]
        mask = (targets != 0).float()
        num_samples = mask.sum()
        
        loss = F.cross_entropy(shifted_logits.view(-1, vocab_size), targets.view(-1), reduce=False)
        loss *= mask.flatten()
        summed_loss = loss.sum()
                
        predicted = shifted_logits.argmax(2)
        num_correct_samples = ((predicted == targets).float()*mask).sum()
        
        return num_samples, num_correct_samples, summed_loss

    return training_step_fn

def train_embeddings2text(model, trainloader, validloader=None, epochs=2, optimizer=None, print_every=1, on_epoch_end=lambda x: x):
    training_step_fn = make_embeddings2text_training_step_fn()
    return train_model(
        model,
        training_step_fn=training_step_fn,
        trainloader=trainloader,
        validloader=validloader,
        epochs=epochs,
        optimizer=optimizer,
        print_every=print_every,
        on_epoch_end=on_epoch_end
    )

def compute_bleu(model,
    input_list,
    target_text_list,
    text_tokenizer,
    input_type='glosses',
    gloss_tokenizer=None,
    eos_token_id=2,
    bos_token_id=1,
    pad_token_id=0,
    print_predictions=False,
    **generate_kwargs
    ):
    
    assert input_type in ['embeddings', 'glosses'], 'input_type can be "glosses" or "embeddings"'
    assert not (input_type == 'glosses' and gloss_tokenizer is None), 'If input_type is "glosses", please provide the gloss_tokenizer'

    metric = datasets.load_metric('bleu')

    all_predictions = []
    all_targets = []

    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for i, (inputs, target_text) in enumerate(zip(tqdm.tqdm(input_list, disable=print_predictions), target_text_list)):
            
            target_ids = text_tokenizer.encode(target_text).ids
            target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)
            
            if input_type == 'glosses':
                input_ids = gloss_tokenizer.encode(inputs).ids
                input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
                inputs_embeds = None
            else:
                assert len(inputs.shape) == 2, 'The input embeddings must have shape (sequence length, embedding dim)'
                input_ids = None
                inputs_embeds = inputs.to(device).unsqueeze(0)

            output_ids = model.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                eos_token_id=eos_token_id,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                length_penalty=1,
                num_beams=5,
                max_length=30,
                **generate_kwargs
            )
            output_ids = output_ids[0]
            
            if print_predictions:
                predicted_sentence = text_tokenizer.decode(output_ids.tolist(), skip_special_tokens=True)
                print(f'{i+1}/{len(input_list)}', predicted_sentence)

            all_predictions.append(output_ids.cpu().numpy().astype(str)[1:-1])
            all_targets.append(target_ids.cpu().unsqueeze(0).numpy().astype(str))

    for i in range(1, 5):
        metric.add_batch(predictions=all_predictions, references=all_targets)
        print(f'BLEU-{i}:', metric.compute(max_order=i)['bleu'])
