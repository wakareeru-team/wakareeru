ALTER TABLE crops ADD COLUMN noise_predicted_prob REAL;
ALTER TABLE crops ADD COLUMN noise_predicted_label TEXT;
ALTER TABLE crops ADD COLUMN noise_prediction_model TEXT;
PRAGMA user_version = 7;
