import tensorflow as tf
import numpy as np

class KnowledgeDistillation:
    def __init__(self, temperature=3.0, alpha=0.1):
        self.temperature = temperature
        self.alpha = alpha
        
    def get_soft_labels(self, teacher_model, data_batch):
        """Get soft labels from teacher model"""
        # Get teacher's logits
        teacher_logits = teacher_model(data_batch)
        # Apply temperature scaling
        soft_targets = tf.nn.softmax(teacher_logits / self.temperature)
        return soft_targets
        
    def distillation_loss(self, y_true, student_logits, teacher_soft_labels):
        """Combined loss: soft targets (KL) and true labels (CE)"""
        # Soft targets loss
        student_prob = tf.nn.softmax(student_logits / self.temperature)
        distillation_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                teacher_soft_labels, student_prob
            )
        )
        
        # Hard targets loss
        student_prob = tf.nn.softmax(student_logits)
        student_loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_true, student_prob)
        )
        
        # Combined loss
        total_loss = (self.alpha * student_loss + 
                     (1 - self.alpha) * distillation_loss)
        
        return total_loss
        
    @tf.function
    def train_step(self, student_model, data_batch, teacher_soft_labels, optimizer):
        """Single training step with knowledge distillation"""
        with tf.GradientTape() as tape:
            # Get student logits
            student_logits = student_model.get_logits(data_batch)
            # Calculate loss
            loss = self.distillation_loss(
                data_batch[1], student_logits, teacher_soft_labels
            )
            
        # Apply gradients
        gradients = tape.gradient(loss, student_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
        
        return loss
