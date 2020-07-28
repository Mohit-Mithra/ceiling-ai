class Seq2seq(keras.layers.Layer):
    """Seq2seq model module
    To do: add docstring to methods
    """
    
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder(y, encoder_outputs, encoder_hidden)
        return decoder_outputs
    def sample_output(self, x, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        output_symbols, _ = self.decoder.forward_sample(encoder_outputs, encoder_hidden)
        return tf.transpose(tf.stack(output_symbols), perm = [0, 1])
        #return return torch.stack(output_symbols).transpose(0,1)   #change line

    def reinforce_forward(self, x, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        self.output_symbols, self.output_logprobs = self.decoder.forward_sample(encoder_outputs, encoder_hidden, reinforce_sample=True)
        return tf.transpose(tf.stack(self.output_symbols), perm = [0, 1])
        #return torch.stack(self.output_symbols).transpose(0,1)  #change line

    def reinforce_backward(self, reward, entropy_factor=0.0):
        assert self.output_logprobs is not None and self.output_symbols is not None, 'must call reinforce_forward first'
        
        with tf.GradientTape(persistent=True) as tape: #Check
            losses = []
            grad_output = []
            for i, symbol in enumerate(self.output_symbols):
                if len(self.output_symbols[0].shape) == 1:
                
                    loss = - tf.math.reduce_sum(torch.diag(torch.index_select(self.output_logprobs[i],1,symbol)),axis=None)*reward + tf.math.reduce_sum(entropy_factor(self.output_logprobs[i]*tf.math.exp(self.output_logprobs[i])),axis=None)
                    
                #loss = - torch.diag(torch.index_select(self.output_logprobs[i], 1, symbol)).sum()*reward \
                #       + entropy_factor*(self.output_logprobs[i]*torch.exp(self.output_logprobs[i])).sum()  #change line #have to do this from scratch I believe.
                else:
                    loss = - self.output_logprobs[i]*reward
                losses.append(loss.sum())
                grad_output.append(None)
         
        tape.gradient(losses, grad_output)
        #torch.autograd.backward(losses, grad_output, retain_graph=True)  #change line
    