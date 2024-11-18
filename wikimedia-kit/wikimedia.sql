CREATE TABLE pagecounts
(
    pagename         VARCHAR(16),
    pageinfo         TEXT, -- 明确指定默认为NULL
    pagecategory     BIGINT      NOT NULL,
    pagecount        BIGINT      NOT NULL                      
);
