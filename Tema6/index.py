import psycopg2

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS equipment_acc;
SET search_path TO equipment_acc;

CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id SERIAL PRIMARY KEY,
    name        VARCHAR(150) NOT NULL UNIQUE,
    tax_id      VARCHAR(20),
    phone       VARCHAR(30),
    email       VARCHAR(100),
    address     TEXT
);

CREATE TABLE IF NOT EXISTS departments (
    department_id SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS workplaces (
    workplace_id  SERIAL PRIMARY KEY,
    department_id INT REFERENCES departments(department_id) ON DELETE SET NULL,
    code          VARCHAR(50) NOT NULL UNIQUE,
    name          VARCHAR(150) NOT NULL,
    location      TEXT
);

CREATE TABLE IF NOT EXISTS employees (
    employee_id   SERIAL PRIMARY KEY,
    last_name     VARCHAR(50) NOT NULL,
    first_name    VARCHAR(50) NOT NULL,
    middle_name   VARCHAR(50),
    position      VARCHAR(100),
    department_id INT REFERENCES departments(department_id) ON DELETE SET NULL,
    is_materially_responsible BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS equipment_types (
    type_id             SERIAL PRIMARY KEY,
    name                VARCHAR(100) NOT NULL UNIQUE,
    useful_life_years   INT,
    description         TEXT
);

CREATE TABLE IF NOT EXISTS purchases (
    purchase_id SERIAL PRIMARY KEY,
    supplier_id INT REFERENCES suppliers(supplier_id) ON DELETE SET NULL,
    doc_number  VARCHAR(50),
    doc_date    DATE NOT NULL,
    total_amount NUMERIC(14,2),
    comments    TEXT
);

CREATE TABLE IF NOT EXISTS equipment (
    equipment_id   SERIAL PRIMARY KEY,
    inventory_number VARCHAR(50) NOT NULL UNIQUE,
    type_id        INT NOT NULL REFERENCES equipment_types(type_id),
    model          VARCHAR(100),
    serial_number  VARCHAR(100),
    purchase_id    INT REFERENCES purchases(purchase_id) ON DELETE SET NULL,
    purchase_date  DATE,
    initial_cost   NUMERIC(14,2),
    current_cost   NUMERIC(14,2),
    supplier_id    INT REFERENCES suppliers(supplier_id) ON DELETE SET NULL,
    current_workplace_id INT REFERENCES workplaces(workplace_id) ON DELETE SET NULL,
    current_responsible_employee_id INT REFERENCES employees(employee_id) ON DELETE SET NULL,
    status         VARCHAR(20) NOT NULL DEFAULT 'in_stock'
        CHECK (status IN ('in_stock','in_use','in_repair','written_off')),
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS equipment_movements (
    movement_id   SERIAL PRIMARY KEY,
    equipment_id  INT NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
    from_workplace_id INT REFERENCES workplaces(workplace_id),
    to_workplace_id   INT REFERENCES workplaces(workplace_id),
    from_responsible_employee_id INT REFERENCES employees(employee_id),
    to_responsible_employee_id   INT REFERENCES employees(employee_id),
    movement_date TIMESTAMP NOT NULL DEFAULT NOW(),
    reason        VARCHAR(200),
    comment       TEXT
);

CREATE TABLE IF NOT EXISTS equipment_revaluations (
    revaluation_id  SERIAL PRIMARY KEY,
    equipment_id    INT NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
    revaluation_date DATE NOT NULL,
    old_cost        NUMERIC(14,2) NOT NULL,
    new_cost        NUMERIC(14,2) NOT NULL,
    reason          TEXT,
    doc_number      VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS equipment_repairs (
    repair_id    SERIAL PRIMARY KEY,
    equipment_id INT NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
    start_date   DATE NOT NULL,
    end_date     DATE,
    description  TEXT,
    repair_cost  NUMERIC(14,2),
    performer    VARCHAR(150),
    status       VARCHAR(20) NOT NULL DEFAULT 'in_progress'
        CHECK (status IN ('planned','in_progress','completed','cancelled'))
);

CREATE INDEX IF NOT EXISTS idx_equipment_type
    ON equipment(type_id);

CREATE INDEX IF NOT EXISTS idx_equipment_current_workplace
    ON equipment(current_workplace_id);

CREATE INDEX IF NOT EXISTS idx_equipment_current_responsible
    ON equipment(current_responsible_employee_id);

CREATE INDEX IF NOT EXISTS idx_movements_equipment
    ON equipment_movements(equipment_id, movement_date);

CREATE INDEX IF NOT EXISTS idx_revaluations_equipment
    ON equipment_revaluations(equipment_id, revaluation_date);

CREATE INDEX IF NOT EXISTS idx_repairs_equipment
    ON equipment_repairs(equipment_id, start_date);
"""

DATA_SQL = """
SET search_path TO equipment_acc;

INSERT INTO suppliers (supplier_id, name, tax_id, phone, email, address) VALUES
(1, 'ТОВ "ТехноСвіт"', '1234567890', '+380441111111', 'info@technosvit.ua', 'м. Київ, вул. Хрещатик, 1')
ON CONFLICT (supplier_id) DO NOTHING;

INSERT INTO suppliers (supplier_id, name, tax_id, phone, email, address) VALUES
(2, 'ТОВ "КомпСервіс"', '0987654321', '+380442222222', 'sales@compservice.ua', 'м. Київ, вул. Сагайдачного, 5')
ON CONFLICT (supplier_id) DO NOTHING;

INSERT INTO departments (department_id, name) VALUES
(1, 'Адміністрація')
ON CONFLICT (department_id) DO NOTHING;

INSERT INTO departments (department_id, name) VALUES
(2, 'Виробничий цех')
ON CONFLICT (department_id) DO NOTHING;

INSERT INTO workplaces (workplace_id, department_id, code, name, location) VALUES
(1, 1, 'OF-01', 'Робоче місце бухгалтера', 'Офіс, каб. 101')
ON CONFLICT (workplace_id) DO NOTHING;

INSERT INTO workplaces (workplace_id, department_id, code, name, location) VALUES
(2, 1, 'OF-02', 'Робоче місце директора', 'Офіс, каб. 102')
ON CONFLICT (workplace_id) DO NOTHING;

INSERT INTO workplaces (workplace_id, department_id, code, name, location) VALUES
(3, 2, 'PR-01', 'Робоче місце оператора верстата', 'Цех 1')
ON CONFLICT (workplace_id) DO NOTHING;

INSERT INTO employees (employee_id, last_name, first_name, middle_name, position, department_id, is_materially_responsible)
VALUES
(1, 'Іваненко', 'Олена', 'Петрівна', 'Бухгалтер', 1, TRUE)
ON CONFLICT (employee_id) DO NOTHING;

INSERT INTO employees (employee_id, last_name, first_name, middle_name, position, department_id, is_materially_responsible)
VALUES
(2, 'Петренко', 'Сергій', 'Іванович', 'Директор', 1, TRUE)
ON CONFLICT (employee_id) DO NOTHING;

INSERT INTO employees (employee_id, last_name, first_name, middle_name, position, department_id, is_materially_responsible)
VALUES
(3, 'Коваленко', 'Микола', 'Олександрович', 'Майстер цеху', 2, TRUE)
ON CONFLICT (employee_id) DO NOTHING;

INSERT INTO equipment_types (type_id, name, useful_life_years, description) VALUES
(1, 'Комп''ютер', 5, 'Персональні комп''ютери та робочі станції')
ON CONFLICT (type_id) DO NOTHING;

INSERT INTO equipment_types (type_id, name, useful_life_years, description) VALUES
(2, 'Верстат токарний', 10, 'Металообробне обладнання')
ON CONFLICT (type_id) DO NOTHING;

INSERT INTO purchases (purchase_id, supplier_id, doc_number, doc_date, total_amount, comments) VALUES
(1, 1, 'INV-001', '2023-01-15', 60000.00, 'Закупівля офісної техніки')
ON CONFLICT (purchase_id) DO NOTHING;

INSERT INTO purchases (purchase_id, supplier_id, doc_number, doc_date, total_amount, comments) VALUES
(2, 2, 'INV-002', '2023-02-10', 150000.00, 'Закупівля токарного верстата')
ON CONFLICT (purchase_id) DO NOTHING;

INSERT INTO equipment (equipment_id, inventory_number, type_id, model, serial_number,
                       purchase_id, purchase_date, initial_cost, current_cost,
                       supplier_id, current_workplace_id, current_responsible_employee_id,
                       status, notes)
VALUES
(1, 'EQ-0001', 1, 'Lenovo ThinkCentre', 'PC123',
 1, '2023-01-15', 20000.00, 20000.00,
 1, 2, 2, 'in_use', 'Робоча станція для офісу')
ON CONFLICT (equipment_id) DO NOTHING;

INSERT INTO equipment (equipment_id, inventory_number, type_id, model, serial_number,
                       purchase_id, purchase_date, initial_cost, current_cost,
                       supplier_id, current_workplace_id, current_responsible_employee_id,
                       status, notes)
VALUES
(2, 'EQ-0002', 1, 'Dell OptiPlex', 'PC124',
 1, '2023-01-15', 20000.00, 20000.00,
 1, NULL, NULL, 'in_stock', 'Резервний комп''ютер')
ON CONFLICT (equipment_id) DO NOTHING;

INSERT INTO equipment (equipment_id, inventory_number, type_id, model, serial_number,
                       purchase_id, purchase_date, initial_cost, current_cost,
                       supplier_id, current_workplace_id, current_responsible_employee_id,
                       status, notes)
VALUES
(3, 'EQ-0003', 2, 'Jupiter-1', 'MACH001',
 2, '2023-02-10', 150000.00, 130000.00,
 2, 3, 3, 'in_use', 'Основний верстат в цеху')
ON CONFLICT (equipment_id) DO NOTHING;

INSERT INTO equipment_movements (movement_id, equipment_id,
    from_workplace_id, to_workplace_id,
    from_responsible_employee_id, to_responsible_employee_id,
    movement_date, reason, comment)
VALUES
(1, 1, NULL, 1, NULL, 1, '2023-01-16 09:00', 'Первинна установка', 'Нове обладнання')
ON CONFLICT (movement_id) DO NOTHING;

INSERT INTO equipment_movements (movement_id, equipment_id,
    from_workplace_id, to_workplace_id,
    from_responsible_employee_id, to_responsible_employee_id,
    movement_date, reason, comment)
VALUES
(2, 1, 1, 2, 1, 2, '2023-06-01 10:00', 'Переміщення', 'Передача робочої станції директору')
ON CONFLICT (movement_id) DO NOTHING;

INSERT INTO equipment_movements (movement_id, equipment_id,
    from_workplace_id, to_workplace_id,
    from_responsible_employee_id, to_responsible_employee_id,
    movement_date, reason, comment)
VALUES
(3, 3, NULL, 3, NULL, 3, '2023-02-15 08:30', 'Первинна установка', 'Введення в експлуатацію')
ON CONFLICT (movement_id) DO NOTHING;

INSERT INTO equipment_revaluations (revaluation_id, equipment_id, revaluation_date,
                                    old_cost, new_cost, reason, doc_number)
VALUES
(1, 3, '2024-01-10', 150000.00, 130000.00, 'Переоцінка через знос', 'RVL-001')
ON CONFLICT (revaluation_id) DO NOTHING;

INSERT INTO equipment_repairs (repair_id, equipment_id, start_date, end_date,
                               description, repair_cost, performer, status)
VALUES
(1, 3, '2024-03-01', '2024-03-15',
 'Заміна підшипників, профілактика', 10000.00, 'ТОВ "РемСервіс"', 'completed')
ON CONFLICT (repair_id) DO NOTHING;
"""

def main():
    conn = psycopg2.connect(
        dbname="equipment_db",   
        user="postgres",        
        password="1111",
        host="localhost",
        port=5432,
    )
    conn.autocommit = True

    with conn.cursor() as cur:
        # Створення структури
        cur.execute(SCHEMA_SQL)
        # Заповнення тестовими даними
        cur.execute(DATA_SQL)

        # Приклад аналітичного запиту:
        # 1) Поточне розміщення устаткування, відповідальний і вартість
        cur.execute("""
            SET search_path TO equipment_acc;
            SELECT
                e.inventory_number,
                e.model,
                et.name AS type_name,
                d.name  AS department_name,
                w.name  AS workplace_name,
                (emp.last_name || ' ' || emp.first_name) AS responsible,
                e.current_cost,
                e.status
            FROM equipment e
            LEFT JOIN equipment_types et
                ON et.type_id = e.type_id
            LEFT JOIN workplaces w
                ON w.workplace_id = e.current_workplace_id
            LEFT JOIN departments d
                ON d.department_id = w.department_id
            LEFT JOIN employees emp
                ON emp.employee_id = e.current_responsible_employee_id
            ORDER BY e.inventory_number;
        """)
        rows = cur.fetchall()

        print("Поточний облік устаткування:")
        for row in rows:
            inv, model, type_name, dept, wp, resp, cost, status = row
            print(f"{inv} | {model} ({type_name}) | "
                  f"Підрозділ: {dept or '-'} | Місце: {wp or '-'} | "
                  f"Відповідальний: {resp or '-'} | Вартість: {cost} | Статус: {status}")

        # 2) Сумарна вартість устаткування по підрозділах і типах
        print("\nСумарна вартість устаткування по підрозділах і типах:")
        cur.execute("""
            SET search_path TO equipment_acc;
            SELECT
                d.name  AS department_name,
                et.name AS type_name,
                SUM(e.current_cost) AS total_cost
            FROM equipment e
            LEFT JOIN equipment_types et
                ON et.type_id = e.type_id
            LEFT JOIN workplaces w
                ON w.workplace_id = e.current_workplace_id
            LEFT JOIN departments d
                ON d.department_id = w.department_id
            WHERE e.status <> 'written_off'
            GROUP BY d.name, et.name
            ORDER BY d.name, et.name;
        """)
        rows = cur.fetchall()
        for dept, type_name, total_cost in rows:
            print(f"{dept or 'Без підрозділу'} | {type_name} | {total_cost}")

    conn.close()

if __name__ == "__main__":
    main()