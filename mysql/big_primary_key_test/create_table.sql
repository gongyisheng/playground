CREATE TABLE 5_big_pk_test (
    big_pk VARCHAR(768),
    big_index_key VARCHAR(768),
    id INT NOT NULL DEFAULT 0,
    PRIMARY KEY (big_pk),
    KEY `query_by_big_key_index` (`big_index_key`)
);