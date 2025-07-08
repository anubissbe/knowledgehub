-- Cosine similarity function for PostgreSQL arrays
-- This function calculates cosine similarity between two double precision arrays

CREATE OR REPLACE FUNCTION cosine_similarity(a double precision[], b double precision[])
RETURNS double precision
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    dot_product double precision := 0;
    norm_a double precision := 0;
    norm_b double precision := 0;
    i integer;
BEGIN
    -- Check if arrays have the same length
    IF array_length(a, 1) != array_length(b, 1) THEN
        RETURN NULL;
    END IF;
    
    -- Calculate dot product and norms
    FOR i IN 1..array_length(a, 1) LOOP
        dot_product := dot_product + (a[i] * b[i]);
        norm_a := norm_a + (a[i] * a[i]);
        norm_b := norm_b + (b[i] * b[i]);
    END LOOP;
    
    -- Calculate cosine similarity
    IF norm_a = 0 OR norm_b = 0 THEN
        RETURN 0;
    END IF;
    
    RETURN dot_product / (sqrt(norm_a) * sqrt(norm_b));
END;
$$;

-- Index for faster similarity searches
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_memories_embedding_gin 
ON memories USING gin(embedding);

-- Alternative function using built-in array operations (potentially faster)
CREATE OR REPLACE FUNCTION cosine_similarity_optimized(a double precision[], b double precision[])
RETURNS double precision
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT 
        CASE 
            WHEN array_length(a, 1) != array_length(b, 1) THEN NULL
            WHEN sqrt(array_sum(array_mult(a, a))) = 0 OR sqrt(array_sum(array_mult(b, b))) = 0 THEN 0
            ELSE array_sum(array_mult(a, b)) / (sqrt(array_sum(array_mult(a, a))) * sqrt(array_sum(array_mult(b, b))))
        END;
$$;

-- Helper functions for array operations
CREATE OR REPLACE FUNCTION array_mult(a double precision[], b double precision[])
RETURNS double precision[]
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT array(
        SELECT a[i] * b[i] 
        FROM generate_subscripts(a, 1) i
        WHERE i <= array_length(b, 1)
    );
$$;

CREATE OR REPLACE FUNCTION array_sum(arr double precision[])
RETURNS double precision
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT sum(x) FROM unnest(arr) x;
$$;