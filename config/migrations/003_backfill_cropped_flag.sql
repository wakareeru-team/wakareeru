UPDATE images SET cropped = 1 WHERE cropped = 0 AND id IN (SELECT DISTINCT image_id FROM crops);
