WITH inv AS (
  SELECT CustomerID, Invoice, MAX(InvoiceDate) AS d, SUM(Revenue) AS v
  FROM sales GROUP BY CustomerID, Invoice),
rfm AS (
  SELECT CustomerID,
         CAST(julianday(:cutoff) - julianday(MAX(d)) AS INTEGER) AS recency_days,
         COUNT(*) AS frequency, SUM(v) AS monetary
  FROM inv GROUP BY CustomerID),
scored AS (
  SELECT *, NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
            NTILE(5) OVER (ORDER BY frequency, monetary)  AS f_score,
            NTILE(5) OVER (ORDER BY monetary)             AS m_score
  FROM rfm)
SELECT *, r_score * 100 + f_score * 10 + m_score AS rfm_code FROM scored;
