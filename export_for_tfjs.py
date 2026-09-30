import tensorflow as tf

model = tf.keras.models.load_model(
    "exported_model/space_object_detector.keras"
)

@tf.function(input_signature=[
    tf.TensorSpec([None, 224, 224, 3], tf.float32)
])
def serving_fn(x):
    return {"output": model(x)}

tf.saved_model.save(
    model,
    "exported_model/tfjs_saved_model",
    signatures={"serving_default": serving_fn}
)

print("EXPORT OK")
