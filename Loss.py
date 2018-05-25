import tensorflow as tf

def loss_drgan(self,adlogits_real,idlogits_real,pslogits_real,content_feature_real,adlogits_fake,idlogits_fake,pslogits_fake,content_feature_fake,output_de_zeros,
               output_de_halves,output_de_ones,poses,labels,fakelabels):

        # ad_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=tf.ones_like(adlogits_real), logits=adlogits_real))
        # ad_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=tf.zeros_like(adlogits_fake), logits=adlogits_fake))




        # with tf.name_scope('pixel-wise_loss'):
        #     output_de_zeros_0,output_de_zeros_1=tf.split(output_de_zeros,2,axis=0)
        #     output_de_ones_0,output_de_ones_1=tf.split(output_de_ones,2,axis=0)
        #     output_de_halves_0,output_de_halves_1=tf.split(output_de_ones,2,axis=0)
        #     gen_loss_L1_0 = tf.reduce_mean(tf.abs(output_de_ones - ))
        #     gen_loss_L1_1 = tf.reduce_mean(tf.abs(targets - outputs))
        #     gen_loss_L1_2 = tf.reduce_mean(tf.abs(targets - outputs))
    # Gloss = ad_loss_syn+id_loss_syn+ps_loss_syn
    # return ad_loss_fake, ad_loss_real, id_loss_real, ps_loss_real, \
    #        ad_loss_syn, id_loss_syn, ps_loss_syn,\
    #         feature_reconstruction_loss,pixel_loss