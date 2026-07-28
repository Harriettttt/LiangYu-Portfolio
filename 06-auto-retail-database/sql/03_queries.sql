
#Query1.Purpose: Display the average price of the car purchased by each customer in the sales records, and sort by price from high to low.
π Customer.Name, AVG(Sale.Price) AS AveragePrice(σ Customer.CustomerID = Sale.CustomerID(Customer ⨝ Sale))
Order by AveragePrice DESC

SELECT Customer.Name, AVG(Sale.Price) AS AveragePrice
FROM Customer
JOIN Sale ON Customer.CustomerID = Sale.CustomerID
GROUP BY Customer.CustomerID, Customer.Name
ORDER BY AveragePrice DESC;

#Query 2:Purpose: Find the dealer with the longest average inventory time.
 Brand.BrandName, AVG(DATEDIFF(CURRENT_DATE, Sale.SaleDate)) AS AverageInventoryTime(σ Brand.BrandID = Model.BrandID ∧ Model.ModelID = Style.ModelID ∧ Style.StyleID = Inventory.StyleID ∧ Inventory.SKU = Sale.SKU(Brand ⨝ Model ⨝ Style ⨝ Inventory ⨝ Sale))
Order by AverageInventoryTime DESC
Limit 1
SELECT Brand.BrandName, AVG(DATEDIFF(CURRENT_DATE, Sale.SaleDate)) AS AverageInventoryTime
FROM Brand
JOIN Model ON Brand.BrandID = Model.BrandID
JOIN Style ON Model.ModelID = Style.ModelID
JOIN Inventory ON Style.StyleID = Inventory.StyleID
JOIN Sale ON Inventory.SKU = Sale.SKU
GROUP BY Brand.BrandID, Brand.BrandName
ORDER BY AverageInventoryTime DESC
LIMIT 1;


#Query 3:Displays sales trends for each brand over the past 3 years by year, month and week. This data is then broken down by buyer's gender and income range.

π b.BrandName AS Brand, YEAR(s.SaleDate) AS SaleYear, MONTH(s.SaleDate) AS SaleMonth, WEEK(s.SaleDate) AS SaleWeek, c.Gender, CASE WHEN c.Income < 30000 THEN 'Low Income' WHEN c.Income >= 30000 AND c.Income < 50000 THEN 'Medium Income' WHEN c.Income >= 50000 THEN 'High Income' END AS IncomeRange, COUNT(*) AS SalesCount(σ s.SaleDate >= DATE_SUB(CURDATE(), INTERVAL 3 YEAR)(Brand ⨝ Model ⨝ Style ⨝ Sale ⨝ Customer))
Group by b.BrandName, YEAR(s.SaleDate), MONTH(s.SaleDate), WEEK(s.SaleDate), c.Gender, IncomeRange
Order by b.BrandName, YEAR(s.SaleDate), MONTH(s.SaleDate), WEEK(s.SaleDate), c.Gender, IncomeRange

SELECT
    b.BrandName AS Brand,
    YEAR(s.SaleDate) AS SaleYear,
    MONTH(s.SaleDate) AS SaleMonth,
    WEEK(s.SaleDate) AS SaleWeek,
    c.Gender,
    CASE
        WHEN c.Income < 30000 THEN 'Low Income'
        WHEN c.Income >= 30000 AND c.Income < 50000 THEN 'Medium Income'
        WHEN c.Income >= 50000 THEN 'High Income'
    END AS IncomeRange,
    COUNT(*) AS SalesCount
FROM
    Brand b
INNER JOIN
    Model m ON b.BrandID = m.BrandID
INNER JOIN
    Style st ON m.ModelID = st.ModelID
INNER JOIN
    Sale s ON st.StyleID = s.SKU
INNER JOIN
    Customer c ON s.CustomerID = c.CustomerID
WHERE
    s.SaleDate >= DATE_SUB(CURDATE(), INTERVAL 3 YEAR)
GROUP BY
    b.BrandName,
    YEAR(s.SaleDate),
    MONTH(s.SaleDate),
    WEEK(s.SaleDate),
    c.Gender,
    IncomeRange
ORDER BY
    b.BrandName,
    YEAR(s.SaleDate),
    MONTH(s.SaleDate),
    WEEK(s.SaleDate),
    c.Gender,
    IncomeRange;

#查询销售日期在特定范围内的销售记录，并按销售日期降序排序：
关系代数：σ(SaleDate >= '开始日期' ∧ SaleDate <= '结束日期') (Sale) ⨝ SaleDate DESC
SELECT SaleID, SaleDate, CustomerID, SKU, Price
FROM Sale
WHERE SaleDate BETWEEN '2023-07-01' AND '2023-12-31'
ORDER BY SaleDate DESC;

#查询所有品牌的名称和对应的型号数量：
关系代数：π BrandName, COUNT(ModelID) (Brand ⨝ Model) ÷ BrandID, BrandName
SELECT b.BrandName, COUNT(m.ModelID) AS ModelCount
FROM Brand b
LEFT JOIN Model m ON b.BrandID = m.BrandID
GROUP BY b.BrandName;