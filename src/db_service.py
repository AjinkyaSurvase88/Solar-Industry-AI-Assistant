import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    # First try os.getenv (for local .env)
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    # If missing, fallback to Streamlit Secrets (for cloud deployment)
    if not all([host, dbname, user, password]):
        try:
            import streamlit as st
            host = st.secrets.get("DB_HOST", host)
            port = st.secrets.get("DB_PORT", port)
            dbname = st.secrets.get("DB_NAME", dbname)
            user = st.secrets.get("DB_USER", user)
            password = st.secrets.get("DB_PASSWORD", password)
        except Exception:
            pass

    if not all([host, dbname, user, password]):
        return None

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            connect_timeout=10
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        try:
            import streamlit as st
            st.error(f"Database connection error: {e}")
        except:
            pass
        return None

def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_analysis_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    city VARCHAR(255),
                    lat FLOAT,
                    lon FLOAT,
                    property_type VARCHAR(100),
                    monthly_bill FLOAT,
                    monthly_consumption FLOAT,
                    roof_area FLOAT,
                    budget FLOAT,
                    elec_rate FLOAT,
                    suitability_score FLOAT,
                    system_size_kwp FLOAT,
                    predicted_generation_kwh FLOAT
                );
            """)
        conn.commit()
        return True
    except Exception as e:
        print(f"Error initializing database table: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def save_analysis_data(data_dict: dict) -> bool:
    """
    Save the user analysis data to the PostgreSQL database.
    data_dict should contain the required keys matching the table columns.
    """
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cur:
            insert_query = """
                INSERT INTO user_analysis_logs (
                    city, lat, lon, property_type, monthly_bill, 
                    monthly_consumption, roof_area, budget, elec_rate, 
                    suitability_score, system_size_kwp, predicted_generation_kwh
                ) VALUES (
                    %(city)s, %(lat)s, %(lon)s, %(property_type)s, %(monthly_bill)s,
                    %(monthly_consumption)s, %(roof_area)s, %(budget)s, %(elec_rate)s,
                    %(suitability_score)s, %(system_size_kwp)s, %(predicted_generation_kwh)s
                );
            """
            cur.execute(insert_query, data_dict)
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving data to database: {e}")
        try:
            import streamlit as st
            st.error(f"Error saving data to Supabase: {e}")
        except:
            pass
        conn.rollback()
        return False
    finally:
        conn.close()