-- Simple cosine similarity function for PostgreSQL arrays
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
    len_a integer;
    len_b integer;
BEGIN
    -- Get array lengths
    len_a := array_length(a, 1);
    len_b := array_length(b, 1);
    
    -- Check if arrays have the same length
    IF len_a IS NULL OR len_b IS NULL OR len_a != len_b THEN
        RETURN 0;
    END IF;
    
    -- Calculate dot product and norms
    FOR i IN 1..len_a LOOP
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