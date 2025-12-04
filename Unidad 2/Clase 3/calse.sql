-- Crear un base de datos
CREATE DATABASE devsenior_db;

-- Usar base de datos
USE devsenior_db;

-- Tabla de Estudiantes
CREATE TABLE estudiantes(
	id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(20) NOT NULL,
    apellido varchar(20) not null,
    edad int not null,
    fecha_registro timestamp default current_timestamp
);

-- Insertar un estudiante 1
insert into estudiantes(nombre, apellido, edad)
value('Carlos','Triana', 29);

-- Insertar un estudiante mas de 1
INSERT INTO estudiantes(nombre, apellido, edad)
VALUES 
('María', 'González', 22),
('Pedro', 'López', 25),
('Ana', 'Martínez', 19),
('Luis', 'Rodríguez', 27),
('Laura', 'Hernández', 23);

-- Consular tabla
select * from estudiantes;

-- Tabla Docente(creada con el workb)
CREATE TABLE `devsenior_db`.`profesores` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `nombre` VARCHAR(45) NOT NULL,
  `apellido` VARCHAR(45) NOT NULL,
  `especialidad` VARCHAR(100) NOT NULL,
  `fecha_creacion` TIMESTAMP NULL DEFAULT current_timestamp,
  PRIMARY KEY (`id`));

-- Insertar un profesor 1
INSERT INTO profesores(nombre, apellido, especialidad)
VALUE
('Juan David', 'Triana', 'Máster IA & Data Science');

-- Insertar un profesor mas de 1
INSERT INTO profesores(nombre, apellido, especialidad)
VALUES 
('Ana María', 'García', 'Desarrollo Web Full Stack'),
('Carlos Eduardo', 'Martínez', 'Base de Datos Avanzadas'),
('Laura Patricia', 'Rodríguez', 'Inteligencia Artificial'),
('Miguel Ángel', 'López', 'Ciberseguridad'),
('Sofia Elena', 'Hernández', 'Machine Learning');

-- Consular tabla
select * from profesores;

-- Tabla Programa
CREATE TABLE programa(
	id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion text,
    profesor_id int not null,
    foreign key (profesor_id) references profesores(id)
);

INSERT INTO programa(nombre, descripcion, profesor_id)
VALUES 
('Máster en Inteligencia Artificial', 'Programa avanzado ', 1),
('Especialización en Data Science', 'Formación en análisis de datos', 1),
('Desarrollo Full Stack', 'Programa completo para desarrollar ', 1),
('Ciberseguridad Avanzada', 'Especialización en seguridad', 1),
('Cloud Computing & DevOps', 'Formación en infraestructura', 1);

select * from programa;

-- Tabla matriculas
CREATE TABLE `devsenior_db`.`matriculas` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `estudiante_id` INT NOT NULL,
  `programa_id` INT NOT NULL,
  `fecha` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  INDEX `programa_id` (`estudiante_id` ASC) INVISIBLE,
  INDEX `programa_id_idx` (`programa_id` ASC) VISIBLE,
  CONSTRAINT `estudiante_id`
    FOREIGN KEY (`estudiante_id`)
    REFERENCES `devsenior_db`.`estudiantes` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `programa_id`
    FOREIGN KEY (`programa_id`)
    REFERENCES `devsenior_db`.`programa` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION);
    
select * from matriculas;
