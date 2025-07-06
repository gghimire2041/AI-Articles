# The Complete Guide to Building Professional Full-Stack React Applications

## Table of Contents

1. [Architecture Overview](#architecture-overview)
1. [Development Environment Setup](#development-environment-setup)
1. [Frontend Development with React](#frontend-development-with-react)
1. [Backend Development with Node.js](#backend-development-with-nodejs)
1. [Database Integration](#database-integration)
1. [Docker Containerization](#docker-containerization)
1. [AI API Integration](#ai-api-integration)
1. [Testing Strategies](#testing-strategies)
1. [Deployment and DevOps](#deployment-and-devops)
1. [Performance Optimization](#performance-optimization)
1. [Security Best Practices](#security-best-practices)
1. [Monitoring and Maintenance](#monitoring-and-maintenance)

-----

## Architecture Overview

### Modern Full-Stack Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Database      │
│   (React)       │◄──►│   (Node.js)     │◄──►│   (MongoDB/     │
│                 │    │                 │    │    PostgreSQL)  │
│ - React 18+     │    │ - Express.js    │    │                 │
│ - TypeScript    │    │ - TypeScript    │    │ - Redis Cache   │
│ - Vite/Webpack  │    │ - JWT Auth      │    │ - Vector DB     │
│ - Tailwind CSS  │    │ - GraphQL/REST  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        │              ┌─────────────────┐              │
        │              │   External      │              │
        └──────────────┤   Services      ├──────────────┘
                       │                 │
                       │ - AI APIs       │
                       │ - Payment APIs  │
                       │ - Email Service │
                       │ - File Storage  │
                       └─────────────────┘
```

### Technology Stack

**Frontend:**

- **React 18+** with Concurrent Features
- **TypeScript** for type safety
- **Vite** for fast development
- **Tailwind CSS** for styling
- **React Query** for data fetching
- **Zustand** for state management

**Backend:**

- **Node.js** with Express.js
- **TypeScript** for consistency
- **GraphQL** or REST APIs
- **JWT** for authentication
- **Prisma** or **Mongoose** for database ORM

**Database:**

- **PostgreSQL** for relational data
- **MongoDB** for document storage
- **Redis** for caching
- **Vector databases** for AI embeddings

**DevOps:**

- **Docker** for containerization
- **Docker Compose** for development
- **Kubernetes** for production
- **CI/CD** with GitHub Actions

-----

## Development Environment Setup

### Prerequisites

```bash
# Install Node.js (v18+)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Project Structure

```
fullstack-react-app/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── types/
│   │   └── utils/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
├── backend/
│   ├── src/
│   │   ├── controllers/
│   │   ├── models/
│   │   ├── routes/
│   │   ├── middleware/
│   │   ├── services/
│   │   └── utils/
│   ├── package.json
│   └── tsconfig.json
├── docker-compose.yml
├── docker-compose.prod.yml
└── README.md
```

### Initial Setup

```bash
# Create project directory
mkdir fullstack-react-app
cd fullstack-react-app

# Initialize frontend
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install

# Additional frontend dependencies
npm install @tanstack/react-query zustand axios
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Initialize backend
cd ../
mkdir backend
cd backend
npm init -y
npm install express cors helmet morgan dotenv
npm install -D @types/node @types/express typescript ts-node nodemon
```

-----

## Frontend Development with React

### Modern React Setup with Vite

**vite.config.ts:**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
    },
  },
})
```

### Component Architecture

**src/components/ui/Button.tsx:**

```typescript
import React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-background',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        outline: 'border border-input hover:bg-accent hover:text-accent-foreground',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'underline-offset-4 hover:underline text-primary',
      },
      size: {
        default: 'h-10 py-2 px-4',
        sm: 'h-9 px-3 rounded-md',
        lg: 'h-11 px-8 rounded-md',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    return (
      <button
        className={buttonVariants({ variant, size, className })}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = 'Button'

export { Button, buttonVariants }
```

### State Management with Zustand

**src/stores/authStore.ts:**

```typescript
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface User {
  id: string
  email: string
  name: string
  role: string
}

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  login: (user: User, token: string) => void
  logout: () => void
  updateUser: (user: Partial<User>) => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      login: (user, token) => {
        set({ user, token, isAuthenticated: true })
      },
      logout: () => {
        set({ user: null, token: null, isAuthenticated: false })
      },
      updateUser: (updates) => {
        const currentUser = get().user
        if (currentUser) {
          set({ user: { ...currentUser, ...updates } })
        }
      },
    }),
    {
      name: 'auth-storage',
    }
  )
)
```

### Data Fetching with React Query

**src/services/api.ts:**

```typescript
import axios from 'axios'
import { useAuthStore } from '@/stores/authStore'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001/api'

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = useAuthStore.getState().token
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout()
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)
```

**src/hooks/useUsers.ts:**

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/services/api'

interface User {
  id: string
  email: string
  name: string
  role: string
  createdAt: string
}

export const useUsers = () => {
  return useQuery({
    queryKey: ['users'],
    queryFn: async () => {
      const response = await api.get<User[]>('/users')
      return response.data
    },
  })
}

export const useCreateUser = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: async (userData: Omit<User, 'id' | 'createdAt'>) => {
      const response = await api.post<User>('/users', userData)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] })
    },
  })
}
```

### Advanced React Patterns

**src/components/DataTable.tsx:**

```typescript
import React, { useMemo } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
  type ColumnFiltersState,
} from '@tanstack/react-table'

interface DataTableProps<TData, TValue> {
  columns: ColumnDef<TData, TValue>[]
  data: TData[]
  onRowClick?: (row: TData) => void
}

export function DataTable<TData, TValue>({
  columns,
  data,
  onRowClick,
}: DataTableProps<TData, TValue>) {
  const [sorting, setSorting] = React.useState<SortingState>([])
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([])

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    state: {
      sorting,
      columnFilters,
    },
  })

  return (
    <div className="rounded-md border">
      <table className="w-full">
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id} className="border-b">
              {headerGroup.headers.map((header) => (
                <th key={header.id} className="px-4 py-3 text-left">
                  {header.isPlaceholder
                    ? null
                    : flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows?.length ? (
            table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className="border-b hover:bg-gray-50 cursor-pointer"
                onClick={() => onRowClick?.(row.original)}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-4 py-3">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={columns.length} className="h-24 text-center">
                No results.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
```

-----

## Backend Development with Node.js

### Express.js Setup with TypeScript

**src/app.ts:**

```typescript
import express from 'express'
import cors from 'cors'
import helmet from 'helmet'
import morgan from 'morgan'
import dotenv from 'dotenv'
import { errorHandler } from './middleware/errorHandler'
import { authRoutes } from './routes/auth'
import { userRoutes } from './routes/users'
import { aiRoutes } from './routes/ai'

dotenv.config()

const app = express()

// Security middleware
app.use(helmet())
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true,
}))

// Logging
app.use(morgan('combined'))

// Body parsing
app.use(express.json({ limit: '10mb' }))
app.use(express.urlencoded({ extended: true }))

// Routes
app.use('/api/auth', authRoutes)
app.use('/api/users', userRoutes)
app.use('/api/ai', aiRoutes)

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() })
})

// Error handling
app.use(errorHandler)

const PORT = process.env.PORT || 3001

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
})
```

### Database Models with Prisma

**prisma/schema.prisma:**

```prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id        String   @id @default(cuid())
  email     String   @unique
  name      String
  password  String
  role      Role     @default(USER)
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  
  posts     Post[]
  comments  Comment[]
  
  @@map("users")
}

model Post {
  id        String   @id @default(cuid())
  title     String
  content   String
  published Boolean  @default(false)
  authorId  String
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  
  author   User      @relation(fields: [authorId], references: [id])
  comments Comment[]
  
  @@map("posts")
}

model Comment {
  id        String   @id @default(cuid())
  content   String
  authorId  String
  postId    String
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  
  author User @relation(fields: [authorId], references: [id])
  post   Post @relation(fields: [postId], references: [id])
  
  @@map("comments")
}

enum Role {
  USER
  ADMIN
}
```

### Authentication Middleware

**src/middleware/auth.ts:**

```typescript
import jwt from 'jsonwebtoken'
import { Request, Response, NextFunction } from 'express'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

interface JwtPayload {
  userId: string
  email: string
  role: string
}

declare global {
  namespace Express {
    interface Request {
      user?: JwtPayload
    }
  }
}

export const authenticate = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '')
    
    if (!token) {
      return res.status(401).json({ error: 'No token provided' })
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as JwtPayload
    
    // Verify user still exists
    const user = await prisma.user.findUnique({
      where: { id: decoded.userId },
      select: { id: true, email: true, role: true }
    })

    if (!user) {
      return res.status(401).json({ error: 'User not found' })
    }

    req.user = decoded
    next()
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' })
  }
}

export const authorize = (roles: string[]) => {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' })
    }

    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Insufficient permissions' })
    }

    next()
  }
}
```

### API Controllers

**src/controllers/userController.ts:**

```typescript
import { Request, Response } from 'express'
import { PrismaClient } from '@prisma/client'
import bcrypt from 'bcrypt'
import { z } from 'zod'

const prisma = new PrismaClient()

const createUserSchema = z.object({
  email: z.string().email(),
  name: z.string().min(2),
  password: z.string().min(8),
  role: z.enum(['USER', 'ADMIN']).optional(),
})

export const createUser = async (req: Request, res: Response) => {
  try {
    const validatedData = createUserSchema.parse(req.body)
    
    // Check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email: validatedData.email }
    })

    if (existingUser) {
      return res.status(400).json({ error: 'User already exists' })
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(validatedData.password, 12)

    // Create user
    const user = await prisma.user.create({
      data: {
        ...validatedData,
        password: hashedPassword,
      },
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        createdAt: true,
      }
    })

    res.status(201).json(user)
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.errors })
    }
    res.status(500).json({ error: 'Internal server error' })
  }
}

export const getUsers = async (req: Request, res: Response) => {
  try {
    const { page = 1, limit = 10, search } = req.query

    const skip = (Number(page) - 1) * Number(limit)

    const where = search ? {
      OR: [
        { name: { contains: search as string, mode: 'insensitive' } },
        { email: { contains: search as string, mode: 'insensitive' } },
      ]
    } : {}

    const [users, total] = await Promise.all([
      prisma.user.findMany({
        where,
        skip,
        take: Number(limit),
        select: {
          id: true,
          email: true,
          name: true,
          role: true,
          createdAt: true,
        },
        orderBy: { createdAt: 'desc' }
      }),
      prisma.user.count({ where })
    ])

    res.json({
      users,
      pagination: {
        page: Number(page),
        limit: Number(limit),
        total,
        pages: Math.ceil(total / Number(limit))
      }
    })
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' })
  }
}
```

### GraphQL Implementation

**src/graphql/schema.ts:**

```typescript
import { buildSchema } from 'graphql'

export const schema = buildSchema(`
  type User {
    id: ID!
    email: String!
    name: String!
    role: Role!
    createdAt: String!
    posts: [Post!]!
  }

  type Post {
    id: ID!
    title: String!
    content: String!
    published: Boolean!
    author: User!
    createdAt: String!
  }

  enum Role {
    USER
    ADMIN
  }

  type Query {
    users(page: Int, limit: Int, search: String): UserConnection!
    user(id: ID!): User
    posts(published: Boolean): [Post!]!
    post(id: ID!): Post
  }

  type Mutation {
    createUser(input: CreateUserInput!): User!
    updateUser(id: ID!, input: UpdateUserInput!): User!
    deleteUser(id: ID!): Boolean!
    createPost(input: CreatePostInput!): Post!
    updatePost(id: ID!, input: UpdatePostInput!): Post!
    deletePost(id: ID!): Boolean!
  }

  type UserConnection {
    users: [User!]!
    pagination: Pagination!
  }

  type Pagination {
    page: Int!
    limit: Int!
    total: Int!
    pages: Int!
  }

  input CreateUserInput {
    email: String!
    name: String!
    password: String!
    role: Role
  }

  input UpdateUserInput {
    email: String
    name: String
    role: Role
  }

  input CreatePostInput {
    title: String!
    content: String!
    published: Boolean
  }

  input UpdatePostInput {
    title: String
    content: String
    published: Boolean
  }
`)
```

-----

## Database Integration

### PostgreSQL with Prisma

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  mongodb:
    image: mongo:6
    environment:
      MONGO_INITDB_ROOT_USERNAME: root
      MONGO_INITDB_ROOT_PASSWORD: password
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

volumes:
  postgres_data:
  redis_data:
  mongodb_data:
```

### MongoDB with Mongoose

**src/models/User.ts:**

```typescript
import mongoose, { Document, Schema } from 'mongoose'

export interface IUser extends Document {
  email: string
  name: string
  password: string
  role: 'USER' | 'ADMIN'
  createdAt: Date
  updatedAt: Date
}

const userSchema = new Schema<IUser>({
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true,
  },
  name: {
    type: String,
    required: true,
    trim: true,
  },
  password: {
    type: String,
    required: true,
    minlength: 8,
  },
  role: {
    type: String,
    enum: ['USER', 'ADMIN'],
    default: 'USER',
  },
}, {
  timestamps: true,
})

userSchema.index({ email: 1 })
userSchema.index({ name: 'text', email: 'text' })

userSchema.methods.toJSON = function() {
  const user = this.toObject()
  delete user.password
  return user
}

export const User = mongoose.model<IUser>('User', userSchema)
```

### Redis for Caching

**src/services/cache.ts:**

```typescript
import Redis from 'ioredis'

class CacheService {
  private redis: Redis

  constructor() {
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: Number(process.env.REDIS_PORT) || 6379,
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3,
    })
  }

  async get<T>(key: string): Promise<T | null> {
    try {
      const value = await this.redis.get(key)
      return value ? JSON.parse(value) : null
    } catch (error) {
      console.error('Cache get error:', error)
      return null
    }
  }

  async set(key: string, value: any, ttl?: number): Promise<void> {
    try {
      const serialized = JSON.stringify(value)
      if (ttl) {
        await this.redis.setex(key, ttl, serialized)
      } else {
        await this.redis.set(key, serialized)
      }
    } catch (error) {
      console.error('Cache set error:', error)
    }
  }

  async del(key: string): Promise<void> {
    try {
      await this.redis.del(key)
    } catch (error) {
      console.error('Cache delete error:', error)
    }
  }

  async exists(key: string): Promise<boolean> {
    try {
      return (await this.redis.exists(key)) === 1
    } catch (error) {
      console.error('Cache exists error:', error)
      return false
    }
  }
}

export const cache = new CacheService()
```

### Database Migrations

**migrations/001_initial_schema.sql:**

```sql
-- Create users table
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255) NOT NULL,
  password VARCHAR(255) NOT NULL,
  role VARCHAR(50) DEFAULT 'USER',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create posts table
CREATE TABLE posts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title VARCHAR(255) NOT NULL,
  content TEXT NOT NULL,
  published BOOLEAN DEFAULT FALSE,
  author_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_posts_author_id ON posts(author_id);
CREATE INDEX idx_posts_published ON posts(published);
CREATE INDEX idx_posts_created_at ON posts(created_at DESC);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_posts_updated_at BEFORE UPDATE ON posts
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

-----

## Docker Containerization

### Multi-stage Dockerfile for Frontend

**frontend/Dockerfile:**

```dockerfile
# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
```

### Backend Dockerfile

**backend/Dockerfile:**

```dockerfile
# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build TypeScript
RUN npm run build

# Production stage
FROM node:18-alpine AS production

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodejs -u 1001

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy built application
COPY --from=build --chown=nodejs:nodejs /app/dist ./dist

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:3001/health || exit 1

# Start application
CMD ["node", "dist/app.js"]
```

### Docker Compose for Development

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "5173:5173"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - VITE_API_URL=http://localhost:3001/api
    depends_on:
      - backend

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    ports:
      - "3001:3001"
    volumes:
      - ./backend:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/myapp
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=your-secret-key
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backend/migrations:/docker-entrypoint-initdb.d
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - frontend
      - backend
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Production Docker Compose

**docker-compose.prod.yml:**

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      - NODE_ENV=production

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    restart: unless-stopped
    environment:
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.prod.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

-----

## AI API Integration

### OpenAI Integration

**src/services/openai.ts:**

```typescript
import OpenAI from 'openai'
import { cache } from './cache'

class OpenAIService {
  private openai: OpenAI

  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    })
  }

  async generateText(prompt: string, options?: {
    model?: string
    maxTokens?: number
    temperature?: number
    useCache?: boolean
  }): Promise<string> {
    const {
      model = 'gpt-3.5-turbo',
      maxTokens = 1000,
      temperature = 0.7,
      useCache = true
    } = options || {}

    // Check cache first
    const cacheKey = `openai:${Buffer.from(prompt).toString('base64')}`
    if (useCache) {
      const cached = await cache.get<string>(cacheKey)
      if (cached) return cached
    }

    try {
      const response = await this.openai.chat.completions.create({
        model,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: maxTokens,
        temperature,
      })

      const result = response.choices[0]?.message?.content || ''
      
      // Cache result for 1 hour
      if (useCache) {
        await cache.set(cacheKey, result, 3600)
      }

      return result
    } catch (error) {
      console.error('OpenAI API error:', error)
      throw new Error('Failed to generate text')
    }
  }

  async generateEmbedding(text: string): Promise<number[]> {
    const cacheKey = `embedding:${Buffer.from(text).toString('base64')}`
    
    const cached = await cache.get<number[]>(cacheKey)
    if (cached) return cached

    try {
      const response = await this.openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: text,
      })

      const embedding = response.data[0].embedding
      
      // Cache embeddings indefinitely
      await cache.set(cacheKey, embedding)
      
      return embedding
    } catch (error) {
      console.error('OpenAI embedding error:', error)
      throw new Error('Failed to generate embedding')
    }
  }

  async moderateContent(content: string): Promise<{
    flagged: boolean
    categories: Record<string, boolean>
  }> {
    try {
      const response = await this.openai.moderations.create({
        input: content,
      })

      const result = response.results[0]
      
      return {
        flagged: result.flagged,
        categories: result.categories,
      }
    } catch (error) {
      console.error('OpenAI moderation error:', error)
      throw new Error('Failed to moderate content')
    }
  }
}

export const openaiService = new OpenAIService()
```

### Vector Database Integration

**src/services/vectordb.ts:**

```typescript
import { PineconeClient } from '@pinecone-database/pinecone'
import { openaiService } from './openai'

class VectorDBService {
  private pinecone: PineconeClient
  private indexName: string

  constructor() {
    this.pinecone = new PineconeClient()
    this.indexName = process.env.PINECONE_INDEX_NAME || 'default'
  }

  async initialize() {
    await this.pinecone.init({
      environment: process.env.PINECONE_ENVIRONMENT!,
      apiKey: process.env.PINECONE_API_KEY!,
    })
  }

  async upsertDocument(id: string, text: string, metadata?: Record<string, any>) {
    const embedding = await openaiService.generateEmbedding(text)
    
    const index = this.pinecone.Index(this.indexName)
    
    await index.upsert({
      upsertRequest: {
        vectors: [{
          id,
          values: embedding,
          metadata: {
            text,
            ...metadata,
          },
        }],
      },
    })
  }

  async similaritySearch(query: string, topK: number = 5): Promise<{
    id: string
    score: number
    metadata: Record<string, any>
  }[]> {
    const queryEmbedding = await openaiService.generateEmbedding(query)
    
    const index = this.pinecone.Index(this.indexName)
    
    const response = await index.query({
      queryRequest: {
        vector: queryEmbedding,
        topK,
        includeMetadata: true,
      },
    })

    return response.matches?.map(match => ({
      id: match.id,
      score: match.score || 0,
      metadata: match.metadata || {},
    })) || []
  }

  async deleteDocument(id: string) {
    const index = this.pinecone.Index(this.indexName)
    
    await index.delete1({
      deleteRequest: {
        ids: [id],
      },
    })
  }
}

export const vectorDBService = new VectorDBService()
```

### AI Controller

**src/controllers/aiController.ts:**

```typescript
import { Request, Response } from 'express'
import { openaiService } from '../services/openai'
import { vectorDBService } from '../services/vectordb'
import { z } from 'zod'

const chatSchema = z.object({
  message: z.string().min(1).max(1000),
  context: z.array(z.string()).optional(),
})

const searchSchema = z.object({
  query: z.string().min(1).max(500),
  limit: z.number().min(1).max(20).optional(),
})

export const chat = async (req: Request, res: Response) => {
  try {
    const { message, context } = chatSchema.parse(req.body)
    
    // Moderate content first
    const moderation = await openaiService.moderateContent(message)
    if (moderation.flagged) {
      return res.status(400).json({
        error: 'Content flagged by moderation',
        categories: moderation.categories,
      })
    }

    // Build prompt with context
    let prompt = message
    if (context && context.length > 0) {
      prompt = `Context: ${context.join('\n')}\n\nQuestion: ${message}`
    }

    const response = await openaiService.generateText(prompt, {
      temperature: 0.7,
      maxTokens: 500,
    })

    res.json({
      response,
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.errors })
    }
    console.error('Chat error:', error)
    res.status(500).json({ error: 'Internal server error' })
  }
}

export const semanticSearch = async (req: Request, res: Response) => {
  try {
    const { query, limit = 5 } = searchSchema.parse(req.body)
    
    const results = await vectorDBService.similaritySearch(query, limit)
    
    res.json({
      query,
      results,
      count: results.length,
    })
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.errors })
    }
    console.error('Semantic search error:', error)
    res.status(500).json({ error: 'Internal server error' })
  }
}

export const indexDocument = async (req: Request, res: Response) => {
  try {
    const { id, text, metadata } = req.body
    
    await vectorDBService.upsertDocument(id, text, metadata)
    
    res.json({
      success: true,
      message: 'Document indexed successfully',
    })
  } catch (error) {
    console.error('Index document error:', error)
    res.status(500).json({ error: 'Internal server error' })
  }
}
```

-----

## Testing Strategies

### Frontend Testing with Vitest

**frontend/vitest.config.ts:**

```typescript
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    globals: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
```

**src/test/setup.ts:**

```typescript
import '@testing-library/jest-dom'
import { beforeAll, afterEach, afterAll } from 'vitest'
import { cleanup } from '@testing-library/react'
import { server } from './mocks/server'

beforeAll(() => server.listen())
afterEach(() => {
  cleanup()
  server.resetHandlers()
})
afterAll(() => server.close())
```

### Component Testing

**src/components/**tests**/Button.test.tsx:**

```typescript
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { Button } from '../ui/Button'

describe('Button', () => {
  it('renders correctly', () => {
    render(<Button>Click me</Button>)
    expect(screen.getByRole('button')).toBeInTheDocument()
    expect(screen.getByText('Click me')).toBeInTheDocument()
  })

  it('calls onClick when clicked', () => {
    const handleClick = vi.fn()
    render(<Button onClick={handleClick}>Click me</Button>)
    
    fireEvent.click(screen.getByRole('button'))
    expect(handleClick).toHaveBeenCalledTimes(1)
  })

  it('applies correct variant classes', () => {
    render(<Button variant="destructive">Delete</Button>)
    expect(screen.getByRole('button')).toHaveClass('bg-destructive')
  })

  it('is disabled when disabled prop is true', () => {
    render(<Button disabled>Disabled</Button>)
    expect(screen.getByRole('button')).toBeDisabled()
  })
})
```

### Backend Testing with Jest

**backend/jest.config.js:**

```javascript
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.test.ts'],
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/app.ts',
  ],
  setupFilesAfterEnv: ['<rootDir>/src/test/setup.ts'],
  testTimeout: 10000,
}
```

**src/test/setup.ts:**

```typescript
import { PrismaClient } from '@prisma/client'
import { execSync } from 'child_process'

const prisma = new PrismaClient()

beforeAll(async () => {
  // Setup test database
  execSync('npx prisma migrate deploy')
  execSync('npx prisma db seed')
})

afterEach(async () => {
  // Clean up test data
  await prisma.comment.deleteMany()
  await prisma.post.deleteMany()
  await prisma.user.deleteMany()
})

afterAll(async () => {
  await prisma.$disconnect()
})
```

### Integration Tests

**src/controllers/**tests**/userController.test.ts:**

```typescript
import request from 'supertest'
import { app } from '../../app'
import { PrismaClient } from '@prisma/client'
import jwt from 'jsonwebtoken'

const prisma = new PrismaClient()

describe('User Controller', () => {
  let authToken: string

  beforeEach(async () => {
    // Create test user
    const user = await prisma.user.create({
      data: {
        email: 'test@example.com',
        name: 'Test User',
        password: 'hashedpassword',
        role: 'ADMIN',
      },
    })

    authToken = jwt.sign(
      { userId: user.id, email: user.email, role: user.role },
      process.env.JWT_SECRET!
    )
  })

  describe('GET /api/users', () => {
    it('should return paginated users', async () => {
      const response = await request(app)
        .get('/api/users')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(response.body).toHaveProperty('users')
      expect(response.body).toHaveProperty('pagination')
      expect(Array.isArray(response.body.users)).toBe(true)
    })

    it('should filter users by search query', async () => {
      await prisma.user.create({
        data: {
          email: 'john@example.com',
          name: 'John Doe',
          password: 'hashedpassword',
        },
      })

      const response = await request(app)
        .get('/api/users?search=john')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(response.body.users).toHaveLength(1)
      expect(response.body.users[0].name).toBe('John Doe')
    })

    it('should return 401 without auth token', async () => {
      await request(app)
        .get('/api/users')
        .expect(401)
    })
  })

  describe('POST /api/users', () => {
    it('should create a new user', async () => {
      const userData = {
        email: 'newuser@example.com',
        name: 'New User',
        password: 'password123',
      }

      const response = await request(app)
        .post('/api/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send(userData)
        .expect(201)

      expect(response.body).toHaveProperty('id')
      expect(response.body.email).toBe(userData.email)
      expect(response.body.name).toBe(userData.name)
      expect(response.body).not.toHaveProperty('password')
    })

    it('should return 400 for invalid data', async () => {
      const response = await request(app)
        .post('/api/users')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ email: 'invalid-email' })
        .expect(400)

      expect(response.body).toHaveProperty('error')
    })
  })
})
```

### E2E Testing with Playwright

**e2e/auth.spec.ts:**

```typescript
import { test, expect } from '@playwright/test'

test.describe('Authentication', () => {
  test('should login successfully', async ({ page }) => {
    await page.goto('/login')
    
    await page.fill('[data-testid="email"]', 'test@example.com')
    await page.fill('[data-testid="password"]', 'password123')
    await page.click('[data-testid="login-button"]')
    
    await expect(page).toHaveURL('/dashboard')
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible()
  })

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login')
    
    await page.fill('[data-testid="email"]', 'invalid@example.com')
    await page.fill('[data-testid="password"]', 'wrongpassword')
    await page.click('[data-testid="login-button"]')
    
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible()
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Invalid credentials')
  })
})
```

-----

## Deployment and DevOps

### CI/CD with GitHub Actions

**.github/workflows/ci-cd.yml:**

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'

    - name: Install dependencies
      run: |
        cd frontend && npm ci
        cd ../backend && npm ci

    - name: Run frontend tests
      run: |
        cd frontend
        npm run test:ci

    - name: Run backend tests
      run: |
        cd backend
        npm run test:ci
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379

    - name: Run E2E tests
      run: |
        npm run e2e:ci
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db

  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3

    - name: Setup Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push frontend
      uses: docker/build-push-action@v4
      with:
        context: ./frontend
        push: true
        tags: |
          myapp/frontend:latest
          myapp/frontend:${{ github.sha }}

    - name: Build and push backend
      uses: docker/build-push-action@v4
      with:
        context: ./backend
        push: true
        tags: |
          myapp/backend:latest
          myapp/backend:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3

    - name: Deploy to production
      uses: appleboy/ssh-action@v0.1.5
      with:
        host: ${{ secrets.HOST }}
        username: ${{ secrets.USERNAME }}
        key: ${{ secrets.SSH_KEY }}
        script: |
          cd /opt/myapp
          git pull origin main
          docker-compose -f docker-compose.prod.yml pull
          docker-compose -f docker-compose.prod.yml up -d
          docker image prune -f
```

### Kubernetes Deployment

**k8s/namespace.yml:**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: myapp
```

**k8s/configmap.yml:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: myapp-config
  namespace: myapp
data:
  NODE_ENV: "production"
  FRONTEND_URL: "https://myapp.com"
  REDIS_URL: "redis://redis-service:6379"
```

**k8s/secrets.yml:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: myapp-secrets
  namespace: myapp
type: Opaque
data:
  jwt-secret: <base64-encoded-jwt-secret>
  database-url: <base64-encoded-database-url>
  openai-api-key: <base64-encoded-openai-key>
```

**k8s/backend-deployment.yml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: myapp/backend:latest
        ports:
        - containerPort: 3001
        env:
        - name: NODE_ENV
          valueFrom:
            configMapKeyRef:
              name: myapp-config
              key: NODE_ENV
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: myapp-secrets
              key: database-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: myapp-secrets
              key: jwt-secret
        livenessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

**k8s/frontend-deployment.yml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: myapp
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: myapp/frontend:latest
        ports:
        - containerPort: 80
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
```

### Monitoring with Prometheus

**k8s/monitoring.yml:**

```yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: myapp-metrics
  namespace: myapp
spec:
  selector:
    matchLabels:
      app: backend
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

-----

## Performance Optimization

### Frontend Optimization

**Code Splitting:**

```typescript
// src/pages/Dashboard.tsx
import { lazy, Suspense } from 'react'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'

const UserManagement = lazy(() => import('../components/UserManagement'))
const Analytics = lazy(() => import('../components/Analytics'))

export const Dashboard = () => {
  return (
    <div className="dashboard">
      <Suspense fallback={<LoadingSpinner />}>
        <UserManagement />
      </Suspense>
      
      <Suspense fallback={<LoadingSpinner />}>
        <Analytics />
      </Suspense>
    </div>
  )
}
```

**Image Optimization:**

```typescript
// src/components/OptimizedImage.tsx
import { useState, useEffect } from 'react'

interface OptimizedImageProps {
  src: string
  alt: string
  className?: string
  width?: number
  height?: number
}

export const OptimizedImage: React.FC<OptimizedImageProps> = ({
  src,
  alt,
  className,
  width,
  height,
}) => {
  const [imageSrc, setImageSrc] = useState<string>('')
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      setImageSrc(src)
      setIsLoaded(true)
    }
    img.src = src
  }, [src])

  return (
    <div className={`relative ${className}`}>
      {!isLoaded && (
        <div className="absolute inset-0 bg-gray-200 animate-pulse rounded" />
      )}
      {imageSrc && (
        <img
          src={imageSrc}
          alt={alt}
          width={width}
          height={height}
          className={`transition-opacity duration-300 ${
            isLoaded ? 'opacity-100' : 'opacity-0'
          }`}
          loading="lazy"
        />
      )}
    </div>
  )
}
```

### Backend Optimization

**Database Query Optimization:**

```typescript
// src/services/userService.ts
import { PrismaClient } from '@prisma/client'
import { cache } from './cache'

const prisma = new PrismaClient()

export class UserService {
  async getUsersWithPosts(page: number = 1, limit: number = 10) {
    const cacheKey = `users:posts:${page}:${limit}`
    
    // Check cache first
    const cached = await cache.get(cacheKey)
    if (cached) return cached

    const skip = (page - 1) * limit

    // Optimized query with select and include
    const users = await prisma.user.findMany({
      skip,
      take: limit,
      select: {
        id: true,
        name: true,
        email: true,
        createdAt: true,
        _count: {
          select: {
            posts: true,
          },
        },
        posts: {
          select: {
            id: true,
            title: true,
            published: true,
            createdAt: true,
          },
          take: 5,
          orderBy: {
            createdAt: 'desc',
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
    })

    // Cache for 5 minutes
    await cache.set(cacheKey, users, 300)
    
    return users
  }

  async getUserById(id: string) {
    const cacheKey = `user:${id}`
    
    const cached = await cache.get(cacheKey)
    if (cached) return cached

    const user = await prisma.user.findUnique({
      where: { id },
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        createdAt: true,
        posts: {
          select: {
            id: true,
            title: true,
            published: true,
            createdAt: true,
          },
          orderBy: {
            createdAt: 'desc',
          },
        },
      },
    })

    if (user) {
      await cache.set(cacheKey, user, 600) // Cache for 10 minutes
    }

    return user
  }
}
```

**API Response Compression:**

```typescript
// src/middleware/compression.ts
import compression from 'compression'
import { Express } from 'express'

export const setupCompression = (app: Express) => {
  app.use(compression({
    level: 6,
    threshold: 1024, // Only compress responses larger than 1KB
    filter: (req, res) => {
      // Don't compress server-sent events
      if (req.headers['accept'] === 'text/event-stream') {
        return false
      }
      return compression.filter(req, res)
    },
  }))
}
```

### Database Optimization

**Indexing Strategy:**

```sql
-- Performance indexes
CREATE INDEX CONCURRENTLY idx_users_email_hash ON users USING hash(email);
CREATE INDEX CONCURRENTLY idx_posts_author_published ON posts(author_id, published);
CREATE INDEX CONCURRENTLY idx_posts_created_at_desc ON posts(created_at DESC);
CREATE INDEX CONCURRENTLY idx_comments_post_created ON comments(post_id, created_at DESC);

-- Partial indexes for common queries
CREATE INDEX CONCURRENTLY idx_posts_published_recent 
ON posts(created_at DESC) 
WHERE published = true;

-- Composite indexes for complex queries
CREATE INDEX CONCURRENTLY idx_users_role_created 
ON users(role, created_at DESC);

-- Full-text search
CREATE INDEX CONCURRENTLY idx_posts_content_fts 
ON posts USING gin(to_tsvector('english', title || ' ' || content));
```

-----

## Security Best Practices

### Authentication & Authorization

**JWT Implementation:**

```typescript
// src/middleware/auth.ts
import jwt from 'jsonwebtoken'
import { Request, Response, NextFunction } from 'express'
import { cache } from '../services/cache'

interface JwtPayload {
  userId: string
  email: string
  role: string
  iat: number
  exp: number
}

export const generateTokens = (payload: Omit<JwtPayload, 'iat' | 'exp'>) => {
  const accessToken = jwt.sign(
    payload,
    process.env.JWT_SECRET!,
    { expiresIn: '15m' }
  )
  
  const refreshToken = jwt.sign(
    payload,
    process.env.JWT_REFRESH_SECRET!,
    { expiresIn: '7d' }
  )

  return { accessToken, refreshToken }
}

export const authenticate = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '')
    
    if (!token) {
      return res.status(401).json({ error: 'No token provided' })
    }

    // Check if token is blacklisted
    const isBlacklisted = await cache.exists(`blacklist:${token}`)
    if (isBlacklisted) {
      return res.status(401).json({ error: 'Token has been revoked' })
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as JwtPayload
    req.user = decoded
    next()
  } catch (error) {
    if (error instanceof jwt.TokenExpiredError) {
      return res.status(401).json({ error: 'Token expired' })
    }
    res.status(401).json({ error: 'Invalid token' })
  }
}

export const logout = async (req: Request, res: Response) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '')
    
    if (token) {
      const decoded = jwt.decode(token) as JwtPayload
      const ttl = decoded.exp - Math.floor(Date.now() / 1000)
      
      if (ttl > 0) {
        await cache.set(`blacklist:${token}`, true, ttl)
      }
    }

    res.json({ message: 'Logged out successfully' })
  } catch (error) {
    res.status(500).json({ error: 'Logout failed' })
  }
}
```

### Input Validation & Sanitization

**Validation Middleware:**

```typescript
// src/middleware/validation.ts
import { z } from 'zod'
import { Request, Response, NextFunction } from 'express'
import DOMPurify from 'isomorphic-dompurify'

export const validate = (schema: z.ZodSchema) => {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      const validatedData = schema.parse(req.body)
      req.body = validatedData
      next()
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({
          error: 'Validation failed',
          details: error.errors,
        })
      }
      res.status(500).json({ error: 'Internal server error' })
    }
  }
}

export const sanitizeHtml = (req: Request, res: Response, next: NextFunction) => {
  const sanitizeObject = (obj: any): any => {
    if (typeof obj === 'string') {
      return DOMPurify.sanitize(obj)
    }
    if (Array.isArray(obj)) {
      return obj.map(sanitizeObject)
    }
    if (obj && typeof obj === 'object') {
      const sanitized: any = {}
      for (const key in obj) {
        sanitized[key] = sanitizeObject(obj[key])
      }
      return sanitized
    }
    return obj
  }

  req.body = sanitizeObject(req.body)
  next()
}
```

### Rate Limiting

**Rate Limiting Middleware:**

```typescript
// src/middleware/rateLimit.ts
import rateLimit from 'express-rate-limit'
import { RedisStore } from 'rate-limit-redis'
import Redis from 'ioredis'

const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: Number(process.env.REDIS_PORT) || 6379,
})

export const createRateLimit = (options: {
  windowMs: number
  max: number
  message?: string
}) => {
  return rateLimit({
    store: new RedisStore({
      client: redis,
      prefix: 'rl:',
    }),
    windowMs: options.windowMs,
    max: options.max,
    message: {
      error: options.message || 'Too many requests',
    },
    standardHeaders: true,
    legacyHeaders: false,
  })
}

// Different rate limits for different endpoints
export const generalRateLimit = createRateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
})

export const authRateLimit = createRateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // limit each IP to 5 requests per windowMs
  message: 'Too many authentication attempts',
})

export const aiRateLimit = createRateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 10, // limit each IP to 10 AI requests per minute
  message: 'AI API rate limit exceeded',
})
```

### Security Headers

**Helmet Configuration:**

```typescript
// src/middleware/security.ts
import helmet from 'helmet'
import { Express } from 'express'

export const setupSecurity = (app: Express) => {
  app.use(helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
        fontSrc: ["'self'", "https://fonts.gstatic.com"],
        scriptSrc: ["'self'"],
        imgSrc: ["'self'", "data:", "https:"],
        connectSrc: ["'self'", "https://api.openai.com"],
        frameSrc: ["'none'"],
        objectSrc: ["'none'"],
        upgradeInsecureRequests: [],
      },
    },
    hsts: {
      maxAge: 31536000,
      includeSubDomains: true,
      preload: true,
    },
    noSniff: true,
    frameguard: { action: 'deny' },
    xssFilter: true,
  }))
}
```

-----

## Monitoring and Maintenance

### Application Monitoring

**Health Checks:**

```typescript
// src/routes/health.ts
import { Router } from 'express'
import { PrismaClient } from '@prisma/client'
import Redis from 'ioredis'

const router = Router()
const prisma = new PrismaClient()
const redis = new Redis()

interface HealthStatus {
  status: 'healthy' | 'unhealthy'
  timestamp: string
  services: {
    database: 'connected' | 'disconnected'
    redis: 'connected' | 'disconnected'
    external: {
      openai: 'available' | 'unavailable'
    }
  }
  metrics: {
    uptime: number
    memory: NodeJS.MemoryUsage
    cpu: number
  }
}

router.get('/health', async (req, res) => {
  const healthStatus: HealthStatus = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    services: {
      database: 'disconnected',
      redis: 'disconnected',
      external: {
        openai: 'unavailable',
      },
    },
    metrics: {
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      cpu: process.cpuUsage().user / 1000000, // Convert to seconds
    },
  }

  // Check database connection
  try {
    await prisma.$queryRaw`SELECT 1`
    healthStatus.services.database = 'connected'
  } catch (error) {
    healthStatus.status = 'unhealthy'
  }

  // Check Redis connection
  try {
    await redis.ping()
    healthStatus.services.redis = 'connected'
  } catch (error) {
    healthStatus.status = 'unhealthy'
  }

  // Check OpenAI API
  try {
    const response = await fetch('https://api.openai.com/v1/models', {
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
      },
    })
    if (response.ok) {
      healthStatus.services.external.openai = 'available'
    }
  } catch (error) {
    // External service failure shouldn't mark the app as unhealthy
  }

  const statusCode = healthStatus.status === 'healthy' ? 200 : 503
  res.status(statusCode).json(healthStatus)
})

export { router as healthRouter }
```

### Logging

**Structured Logging:**

```typescript
// src/utils/logger.ts
import winston from 'winston'

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'myapp-backend' },
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
  ],
})

if (process.env.NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.simple()
  }))
}

export { logger }
```

### Error Tracking

**Error Handler:**

```typescript
// src/middleware/errorHandler.ts
import { Request, Response, NextFunction } from 'express'
import { logger } from '../utils/logger'

export class AppError extends Error {
  statusCode: number
  isOperational: boolean

  constructor(message: string, statusCode: number) {
    super(message)
    this.statusCode = statusCode
    this.isOperational = true

    Error.captureStackTrace(this, this.constructor)
  }
}

export const errorHandler = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  let error = { ...err }
  error.message = err.message

  // Log error
  logger.error({
    message: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent'),
  })

  // Mongoose bad ObjectId
  if (err.name === 'CastError') {
    const message = 'Resource not found'
    error = new AppError(message, 404)
  }

  // Mongoose duplicate key
  if (err.name === 'MongoError' && (err as any).code === 11000) {
    const message = 'Duplicate field value entered'
    error = new AppError(message, 400)
  }

  // Mongoose validation error
  if (err.name === 'ValidationError') {
    const message = Object.values((err as any).errors).map((val: any) => val.message)
    error = new AppError(message.join(', '), 400)
  }

  res.status((error as AppError).statusCode || 500).json({
    success: false,
    error: error.message || 'Server Error',
  })
}
```

### Performance Monitoring

**Metrics Collection:**

```typescript
// src/middleware/metrics.ts
import { Request, Response, NextFunction } from 'express'
import { register, Counter, Histogram, collectDefaultMetrics } from 'prom-client'

// Collect default metrics
collectDefaultMetrics()

// Custom metrics
const httpRequestsTotal = new Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code'],
})

const httpRequestDuration = new Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.1, 0.5, 1, 2, 5],
})

export const metricsMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const start = Date.now()

  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000
    const route = req.route?.path || req.path
    
    httpRequestsTotal.inc({
      method: req.method,
      route,
      status_code: res.statusCode,
    })

    httpRequestDuration.observe(
      {
        method: req.method,
        route,
        status_code: res.statusCode,
      },
      duration
    )
  })

  next()
}

export const metricsEndpoint = (req: Request, res: Response) => {
  res.set('Content-Type', register.contentType)
  res.end(register.metrics())
}
```

-----

## Conclusion

This comprehensive guide covers the essential aspects of building professional full-stack React applications. From setting up the development environment to deploying production-ready applications, you now have the knowledge to create scalable, secure, and performant web applications.

### Key Takeaways:

1. **Architecture**: Use a modular, scalable architecture with clear separation of concerns
1. **TypeScript**: Implement type safety across the entire stack
1. **Database**: Choose the right database for your use case and optimize queries
1. **Security**: Implement proper authentication, authorization, and input validation
1. **Testing**: Write comprehensive tests at unit, integration, and E2E levels
1. **Performance**: Optimize both frontend and backend for maximum efficiency
1. **Monitoring**: Implement proper logging, error tracking, and performance monitoring
1. **DevOps**: Use containerization and CI/CD for reliable deployments

### Next Steps:

1. Start with a simple project implementing the core concepts
1. Gradually add complexity as you become more comfortable
1. Focus on one area at a time (frontend, backend, DevOps)
1. Join the React and Node.js communities for support
1. Stay updated with the latest best practices and technologies

Remember that building full-stack applications is an iterative process. Start simple, test frequently, and continuously refine your approach based on real-world feedback and performance metrics.

-----

*This guide provides a comprehensive foundation for building modern full-stack React applications. Each section can be expanded based on specific project requirements and constraints.*
