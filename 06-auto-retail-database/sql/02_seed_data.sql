INSERT INTO Brand (BrandName) VALUES ('Toyota');
INSERT INTO Brand (BrandName) VALUES ('Honda');
INSERT INTO Brand (BrandName) VALUES ('Ford');
INSERT INTO Brand (BrandName) VALUES ('Chevrolet');
INSERT INTO Brand (BrandName) VALUES ('Nissan');
INSERT INTO Brand (BrandName) VALUES ('BMW');
INSERT INTO Brand (BrandName) VALUES ('Mercedes-Benz');
INSERT INTO Brand (BrandName) VALUES ('Audi');
INSERT INTO Brand (BrandName) VALUES ('Volkswagen');
INSERT INTO Brand (BrandName) VALUES ('Hyundai');

INSERT INTO Model (ModelName, BrandID) VALUES ('Camry', 1);
INSERT INTO Model (ModelName, BrandID) VALUES ('Civic', 2);
INSERT INTO Model (ModelName, BrandID) VALUES ('Focus', 3);
INSERT INTO Model (ModelName, BrandID) VALUES ('Silverado', 4);
INSERT INTO Model (ModelName, BrandID) VALUES ('Altima', 5);
INSERT INTO Model (ModelName, BrandID) VALUES ('3 Series', 6);
INSERT INTO Model (ModelName, BrandID) VALUES ('E-Class', 7);
INSERT INTO Model (ModelName, BrandID) VALUES ('A4', 8);
INSERT INTO Model (ModelName, BrandID) VALUES ('Golf', 9);
INSERT INTO Model (ModelName, BrandID) VALUES ('Elantra', 10);

INSERT INTO Style (StyleName, Price, ModelID) VALUES ('SE', 25000, 1);
INSERT INTO Style (StyleName, Price, ModelID) VALUES ('EX', 27000, 2);
INSERT INTO Style (StyleName, Price, ModelID) VALUES ('Titanium', 28000, 3);
INSERT INTO Style (StyleName, Price, ModelID) VALUES ('LTZ', 35000, 4);
INSERT INTO Style (StyleName, Price, ModelID) VALUES ('SL', 29000, 5);
INSERT INTO Style (StyleName, Price, ModelID) VALUES ('330i', 40000, 6);
INSERT INTO Style (StyleName, Price, ModelID) VALUES ('E350', 50000, 7);
INSERT INTO Style (StyleName, Price, ModelID) VALUES ('Premium', 45000, 8);
INSERT INTO Style (StyleName, Price, ModelID) VALUES ('GLI', 32000, 9);
INSERT INTO Style (StyleName, Price, ModelID) VALUES ('Limited', 26000, 10);


INSERT INTO Part (PartName, SupplierName) VALUES ('Engine', 'ABC Supplier');
INSERT INTO Part (PartName, SupplierName) VALUES ('Brake Pad', 'XYZ Supplier');
INSERT INTO Part (PartName, SupplierName) VALUES ('Radiator', '123 Supplier');
INSERT INTO Part (PartName, SupplierName) VALUES ('Headlight', '456 Supplier');
INSERT INTO Part (PartName, SupplierName) VALUES ('Battery', 'DEF Supplier');
INSERT INTO Part (PartName, SupplierName) VALUES ('Alternator', '789 Supplier');
INSERT INTO Part (PartName, SupplierName) VALUES ('Tire', 'GHI Supplier');
INSERT INTO Part (PartName, SupplierName) VALUES ('Shock Absorber', 'JKL Supplier');
INSERT INTO Part (PartName, SupplierName) VALUES ('Air Filter', 'MNO Supplier');
INSERT INTO Part (PartName, SupplierName) VALUES ('Oil Filter', 'PQR Supplier');

INSERT INTO Customer (Name, Address, Phone, Gender, Income) VALUES ('John Smith', '123 Main St', '555-1234', 'M', 50000);
INSERT INTO Customer (Name, Address, Phone, Gender, Income) VALUES ('Jane Doe', '456 Elm St', '555-5678', 'F', 60000);
INSERT INTO Customer (Name, Address, Phone, Gender, Income) VALUES ('Mike Johnson', '789 Oak St', '555-9876', 'M', 75000);
INSERT INTO Customer (Name, Address, Phone, Gender, Income) VALUES ('Emily Davis', '321 Pine St', '555-4321', 'F', 55000);
INSERT INTO Customer (Name, Address, Phone, Gender, Income) VALUES ('David Wilson', '654 Birch St', '555-8765', 'M', 80000);
INSERT INTO Customer (Name, Address, Phone, Gender, Income) VALUES ('Sarah Thompson', '987 Maple St', '555-2345', 'F', 65000);
INSERT INTO Customer (Name, Address, Phone, Gender, Income) VALUES ('Michael Brown', '159 Cedar St', '555-7654', 'M', 70000);
INSERT INTO Customer (Name, Address, Phone, Gender, Income) VALUES ('Jessica Taylor', '753 Walnut St', '555-3456', 'F', 60000);
INSERT INTO Customer (Name, Address, Phone, Gender, Income) VALUES ('Andrew Miller', '246 Pine St', '555-6543', 'M', 55000);
INSERT INTO Customer (Name, Address, Phone, Gender, Income) VALUES ('Olivia Anderson', '864 Oak St', '555-4567', 'F', 70000);

INSERT INTO Inventory (Quantity, StyleID) VALUES (5, 1);
INSERT INTO Inventory (Quantity, StyleID) VALUES (10, 2);
INSERT INTO Inventory (Quantity, StyleID) VALUES (7, 3);
INSERT INTO Inventory (Quantity, StyleID) VALUES (3, 4);
INSERT INTO Inventory (Quantity, StyleID) VALUES (8, 5);
INSERT INTO Inventory (Quantity, StyleID) VALUES (2, 6);
INSERT INTO Inventory (Quantity, StyleID) VALUES (6, 7);
INSERT INTO Inventory (Quantity, StyleID) VALUES (4, 8);
INSERT INTO Inventory (Quantity, StyleID) VALUES (9, 9);
INSERT INTO Inventory (Quantity, StyleID) VALUES (1, 10);

INSERT INTO Purchase (PurchaseDate, Quantity, StyleID, PartID) VALUES ('2023-01-01', 2, 1, 1);
INSERT INTO Purchase (PurchaseDate, Quantity, StyleID, PartID) VALUES ('2023-02-02', 3, 2, 2);
INSERT INTO Purchase (PurchaseDate, Quantity, StyleID, PartID) VALUES ('2023-03-03', 1, 3, 3);
INSERT INTO Purchase (PurchaseDate, Quantity, StyleID, PartID) VALUES ('2023-04-04', 4, 4, 4);
INSERT INTO Purchase (PurchaseDate, Quantity, StyleID, PartID) VALUES ('2023-05-05', 2, 5, 5);
INSERT INTO Purchase (PurchaseDate, Quantity, StyleID, PartID) VALUES ('2023-06-06', 1, 6, 6);
INSERT INTO Purchase (PurchaseDate, Quantity, StyleID, PartID) VALUES ('2023-07-07', 3, 7, 7);
INSERT INTO Purchase (PurchaseDate, Quantity, StyleID, PartID) VALUES ('2023-08-08', 2, 8, 8);
INSERT INTO Purchase (PurchaseDate, Quantity, StyleID, PartID) VALUES ('2023-09-09', 4, 9, 9);
INSERT INTO Purchase (PurchaseDate, Quantity, StyleID, PartID) VALUES ('2023-10-10', 1, 10, 10);

INSERT INTO Sale (SaleDate, CustomerID, SKU, Price) VALUES ('2023-01-01', 1, 1, 26000);
INSERT INTO Sale (SaleDate, CustomerID, SKU, Price) VALUES ('2023-02-02', 2, 2, 28000);
INSERT INTO Sale (SaleDate, CustomerID, SKU, Price) VALUES ('2023-03-03', 3, 3, 29000);
INSERT INTO Sale (SaleDate, CustomerID, SKU, Price) VALUES ('2023-04-04', 4, 4, 35000);
INSERT INTO Sale (SaleDate, CustomerID, SKU, Price) VALUES ('2023-05-05', 5, 5, 30000);
INSERT INTO Sale (SaleDate, CustomerID, SKU, Price) VALUES ('2023-06-06', 6, 6, 40000);
INSERT INTO Sale (SaleDate, CustomerID, SKU, Price) VALUES ('2023-07-07', 7, 7, 50000);
INSERT INTO Sale (SaleDate, CustomerID, SKU, Price) VALUES ('2023-08-08', 8, 8, 45000);
INSERT INTO Sale (SaleDate, CustomerID, SKU, Price) VALUES ('2023-09-09', 9, 9, 32000);
INSERT INTO Sale (SaleDate, CustomerID, SKU, Price) VALUES ('2023-10-10', 10, 10, 26000);

INSERT INTO Maintenance (VIN, MaintenanceDate, OperatorRecord) VALUES ('123ABC', '2023-01-01', 'Performed routine maintenance');
INSERT INTO Maintenance (VIN, MaintenanceDate, OperatorRecord) VALUES ('456DEF', '2023-02-02', 'Replaced brake pads');
INSERT INTO Maintenance (VIN, MaintenanceDate, OperatorRecord) VALUES ('789GHI', '2023-03-03', 'Changed oil and filter');
INSERT INTO Maintenance (VIN, MaintenanceDate, OperatorRecord) VALUES ('ABC123', '2023-04-04', 'Repaired exhaust system');
INSERT INTO Maintenance (VIN, MaintenanceDate, OperatorRecord) VALUES ('DEF456', '2023-05-05', 'Replaced battery');
INSERT INTO Maintenance (VIN, MaintenanceDate, OperatorRecord) VALUES ('GHI789', '2023-06-06', 'Performed tire rotation');
INSERT INTO Maintenance (VIN, MaintenanceDate, OperatorRecord) VALUES ('123XYZ', '2023-07-07', 'Fixed electrical issue');
INSERT INTO Maintenance (VIN, MaintenanceDate, OperatorRecord) VALUES ('456ABC', '2023-08-08', 'Replaced spark plugs');
INSERT INTO Maintenance (VIN, MaintenanceDate, OperatorRecord) VALUES ('789DEF', '2023-09-09', 'Repaired cooling system');
INSERT INTO Maintenance (VIN, MaintenanceDate, OperatorRecord) VALUES ('XYZ123', '2023-10-10', 'Performed engine tune-up');





